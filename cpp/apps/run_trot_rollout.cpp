#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "go2wbc/CentroidalMpc.hpp"
#include "go2wbc/GeneralContactWbc.hpp"

using go2wbc::CentroidalMpc;
using go2wbc::CentroidalMpcConfig;
using go2wbc::CentroidalMpcInput;
using go2wbc::Foot;
using go2wbc::FOOT_FL;
using go2wbc::FOOT_FR;
using go2wbc::FOOT_RL;
using go2wbc::FOOT_RR;
using go2wbc::GeneralContactWbc;
using go2wbc::GeneralContactWbcConfig;
using go2wbc::GeneralContactWbcInput;
using go2wbc::MatrixX;
using go2wbc::MujocoModelInterface;
using go2wbc::Vector3;
using go2wbc::VectorX;

struct TrotWindow {
    std::array<Foot, 2> swing_feet;
    double start;
    double duration;

    double end() const { return start + duration; }
};

struct SwingPlan {
    Foot foot;
    Vector3 start_position;
    Vector3 target_position;
};

struct SwingReference {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
};

struct CommandSegment {
    double duration;
    double vx;
    double vy;
    double yaw_rate;
};

struct TimerStats {
    double mpc_ms;
    double wbc_ms;
    double step_ms;
    int mpc_count;
    int wbc_count;
    int step_count;

    TimerStats() : mpc_ms(0.0), wbc_ms(0.0), step_ms(0.0), mpc_count(0), wbc_count(0), step_count(0) {}
};

const double kPi = 3.14159265358979323846;

CommandSegment commandAt(const std::vector<CommandSegment>& segments, double command_time, double* elapsed_before) {
    double elapsed = 0.0;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (command_time < elapsed + segments[i].duration) {
            *elapsed_before = elapsed;
            return segments[i];
        }
        elapsed += segments[i].duration;
    }
    *elapsed_before = elapsed;
    CommandSegment stop;
    stop.duration = 1.0;
    stop.vx = 0.0;
    stop.vy = 0.0;
    stop.yaw_rate = 0.0;
    return stop;
}

Vector3 integratedCommandPose(const std::vector<CommandSegment>& segments, double command_time) {
    Vector3 pose = Vector3::Zero();  // x, y, yaw
    double elapsed = 0.0;
    for (size_t i = 0; i < segments.size(); ++i) {
        double dt = std::min(segments[i].duration, std::max(0.0, command_time - elapsed));
        if (dt <= 0.0) {
            break;
        }
        pose(0) += segments[i].vx * dt;
        pose(1) += segments[i].vy * dt;
        pose(2) += segments[i].yaw_rate * dt;
        elapsed += segments[i].duration;
    }
    return pose;
}

double totalCommandDuration(const std::vector<CommandSegment>& segments) {
    double total = 0.0;
    for (size_t i = 0; i < segments.size(); ++i) {
        total += segments[i].duration;
    }
    return total;
}

double nowMs() {
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point start = Clock::now();
    std::chrono::duration<double, std::milli> elapsed = Clock::now() - start;
    return elapsed.count();
}

double maxAbs(const VectorX& v) {
    if (v.size() == 0) {
        return 0.0;
    }
    return v.cwiseAbs().maxCoeff();
}

Vector3 limitedPlanarDelta(const Vector3& delta, double max_step_length) {
    Vector3 out = delta;
    double norm_xy = std::sqrt(out(0) * out(0) + out(1) * out(1));
    if (norm_xy > max_step_length && norm_xy > 0.0) {
        out(0) *= max_step_length / norm_xy;
        out(1) *= max_step_length / norm_xy;
    }
    return out;
}

std::vector<TrotWindow> buildWindows(int cycles, double swing_duration, double stance_gap) {
    std::vector<TrotWindow> windows;
    double start = 1.0;
    double stride = swing_duration + stance_gap;
    for (int i = 0; i < 2 * cycles; ++i) {
        TrotWindow window;
        window.swing_feet = (i % 2 == 0)
            ? std::array<Foot, 2>{{FOOT_FL, FOOT_RR}}
            : std::array<Foot, 2>{{FOOT_FR, FOOT_RL}};
        window.start = start + static_cast<double>(i) * stride;
        window.duration = swing_duration;
        windows.push_back(window);
    }
    return windows;
}

bool containsFoot(const std::array<Foot, 2>& feet, Foot foot) {
    return feet[0] == foot || feet[1] == foot;
}

Vector3 footholdDeltaForFoot(
    Foot foot,
    const std::array<Vector3, go2wbc::kNumFeet>& initial_feet,
    const Vector3& step_delta,
    double yaw_delta,
    double max_step_length
) {
    Vector3 delta = step_delta;
    if (yaw_delta != 0.0) {
        Vector3 center = Vector3::Zero();
        for (Foot f : go2wbc::allFeet()) {
            center += initial_feet[static_cast<size_t>(f)];
        }
        center /= static_cast<double>(go2wbc::kNumFeet);
        Vector3 offset = initial_feet[static_cast<size_t>(foot)] - center;
        delta(0) += yaw_delta * (-offset(1));
        delta(1) += yaw_delta * offset(0);
    }
    return limitedPlanarDelta(delta, max_step_length);
}

void smoothstep(double r, double* s, double* ds, double* dds) {
    if (r < 0.0) {
        r = 0.0;
    }
    if (r > 1.0) {
        r = 1.0;
    }
    *s = 3.0 * r * r - 2.0 * r * r * r;
    *ds = 6.0 * r - 6.0 * r * r;
    *dds = 6.0 - 12.0 * r;
}

SwingReference swingReference(
    const Vector3& p0,
    const Vector3& delta,
    double height,
    double start,
    double duration,
    double time
) {
    SwingReference ref;
    ref.position = p0;
    ref.velocity = Vector3::Zero();
    ref.acceleration = Vector3::Zero();
    if (time <= start) {
        return ref;
    }
    if (time >= start + duration) {
        ref.position = p0 + delta;
        return ref;
    }

    double r = (time - start) / duration;
    double s, ds_dr, dds_dr2;
    smoothstep(r, &s, &ds_dr, &dds_dr2);
    double sdot = ds_dr / duration;
    double sddot = dds_dr2 / (duration * duration);

    ref.position = p0 + delta * s;
    ref.velocity = delta * sdot;
    ref.acceleration = delta * sddot;

    double sin_term = std::sin(kPi * s);
    double cos_term = std::cos(kPi * s);
    ref.position(2) += height * sin_term;
    ref.velocity(2) += height * kPi * cos_term * sdot;
    ref.acceleration(2) += height * (-kPi * kPi * sin_term * sdot * sdot + kPi * cos_term * sddot);
    return ref;
}

void setYawQuat(VectorX& qpos, double yaw) {
    double half = 0.5 * yaw;
    qpos(3) = std::cos(half);
    qpos(4) = 0.0;
    qpos(5) = 0.0;
    qpos(6) = std::sin(half);
}

VectorX footCenteredBaseReference(
    const VectorX& home_qpos,
    const Vector3& initial_base,
    const std::array<Vector3, go2wbc::kNumFeet>& initial_feet,
    const std::array<Vector3, go2wbc::kNumFeet>& planned_feet,
    double yaw
) {
    VectorX qref = home_qpos;
    Vector3 mean_delta = Vector3::Zero();
    for (Foot foot : go2wbc::allFeet()) {
        mean_delta += planned_feet[static_cast<size_t>(foot)] - initial_feet[static_cast<size_t>(foot)];
    }
    mean_delta /= static_cast<double>(go2wbc::kNumFeet);
    qref(0) = initial_base(0) + mean_delta(0);
    qref(1) = initial_base(1) + mean_delta(1);
    setYawQuat(qref, yaw);
    return qref;
}

std::vector<Foot> stanceFeetForWindow(const TrotWindow* window) {
    std::vector<Foot> stance;
    for (Foot foot : go2wbc::allFeet()) {
        if (window == 0 || !containsFoot(window->swing_feet, foot)) {
            stance.push_back(foot);
        }
    }
    return stance;
}

std::vector<Foot> swingFeetForWindow(const TrotWindow* window) {
    std::vector<Foot> swing;
    if (window != 0) {
        swing.push_back(window->swing_feet[0]);
        swing.push_back(window->swing_feet[1]);
    }
    return swing;
}

CentroidalMpcInput makeMpcInput(
    MujocoModelInterface& robot,
    const CentroidalMpcConfig& cfg,
    const std::vector<TrotWindow>& windows,
    const TrotWindow* active_window,
    double sim_time,
    const Vector3& com_ref,
    const Vector3& com_vel_ref,
    const Vector3& ori_ref,
    const Vector3& omega_ref
) {
    CentroidalMpcInput input;
    input.com_position_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.com_velocity_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.orientation_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.angular_velocity_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    for (int k = 0; k <= cfg.horizon_steps; ++k) {
        input.com_position_ref.row(k) = com_ref.transpose();
        input.com_velocity_ref.row(k) = com_vel_ref.transpose();
        input.orientation_ref.row(k) = ori_ref.transpose();
        input.angular_velocity_ref.row(k) = omega_ref.transpose();
    }

    input.contact_schedule.resize(static_cast<size_t>(cfg.horizon_steps));
    for (int k = 0; k < cfg.horizon_steps; ++k) {
        double knot_time = sim_time + cfg.dt * static_cast<double>(k);
        input.contact_schedule[static_cast<size_t>(k)].fill(true);
        for (size_t i = 0; i < windows.size(); ++i) {
            const TrotWindow& window = windows[i];
            if (window.start <= knot_time && knot_time < window.end()) {
                input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(window.swing_feet[0])] = false;
                input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(window.swing_feet[1])] = false;
            }
        }
        if (active_window != 0) {
            input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(active_window->swing_feet[0])] = false;
            input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(active_window->swing_feet[1])] = false;
        }
    }
    (void)robot;
    return input;
}

VectorX forceRefForFeet(const VectorX& all_forces, const std::vector<Foot>& feet) {
    VectorX out(3 * static_cast<int>(feet.size()));
    for (int i = 0; i < static_cast<int>(feet.size()); ++i) {
        int foot_id = static_cast<int>(feet[static_cast<size_t>(i)]);
        out.segment(3 * i, 3) = all_forces.segment(3 * foot_id, 3);
    }
    return out;
}

bool isSolved(const std::string& status) {
    return status == "solved" || status == "solved inaccurate";
}

void writeCsvHeader(std::ofstream& stream, int nq, int nv) {
    stream << "time";
    for (int i = 0; i < nq; ++i) {
        stream << ",qpos" << i;
    }
    for (int i = 0; i < nv; ++i) {
        stream << ",qvel" << i;
    }
    stream << "\n";
}

void writeCsvSample(std::ofstream& stream, const MujocoModelInterface& robot) {
    stream << std::fixed << std::setprecision(10) << robot.data()->time;
    for (int i = 0; i < robot.nq(); ++i) {
        stream << "," << robot.data()->qpos[i];
    }
    for (int i = 0; i < robot.nv(); ++i) {
        stream << "," << robot.data()->qvel[i];
    }
    stream << "\n";
}

int main(int argc, char** argv) {
    try {
        std::string model_path = ".\\models\\mujoco_menagerie\\unitree_go2\\scene.xml";
        std::string record_csv_path;
        double vx = 0.012;
        double vy = 0.0;
        double yaw_rate = 0.0;
        int cycles = 3;
        double swing_duration = 0.35;
        double stance_gap = 0.45;
        double swing_height = 0.035;
        double max_step_length = 0.035;
        std::vector<CommandSegment> command_segments;
        if (argc >= 2) {
            model_path = argv[1];
        }
        bool route_mode = argc >= 3 && std::string(argv[2]) == "route";
        if (route_mode) {
            CommandSegment forward1 = {12.0, 0.040, 0.0, 0.0};
            CommandSegment turn = {20.0, 0.004, 0.0, kPi / 40.0};
            CommandSegment forward2 = {12.0, 0.040, 0.0, 0.0};
            CommandSegment stop = {2.0, 0.0, 0.0, 0.0};
            command_segments.push_back(forward1);
            command_segments.push_back(turn);
            command_segments.push_back(forward2);
            command_segments.push_back(stop);
            vx = 0.040;
            yaw_rate = 0.0;
            if (argc >= 4) {
                record_csv_path = argv[3];
            }
        } else if (argc >= 3) {
            vx = std::atof(argv[2]);
            if (argc >= 4) {
                yaw_rate = std::atof(argv[3]);
            }
            if (argc >= 5) {
                record_csv_path = argv[4];
            }
            if (argc >= 6) {
                cycles = std::max(1, std::atoi(argv[5]));
            }
        }
        if (command_segments.empty()) {
            CommandSegment constant = {2.0 * cycles * (swing_duration + stance_gap), vx, vy, yaw_rate};
            command_segments.push_back(constant);
        }
        double command_duration = totalCommandDuration(command_segments);
        if (route_mode) {
            cycles = std::max(1, static_cast<int>(std::ceil(command_duration / (2.0 * (swing_duration + stance_gap)))));
        }

        MujocoModelInterface robot(model_path);
        robot.setKeyframe("home");

        VectorX home_qpos = robot.qpos();
        Vector3 home_com = robot.centerOfMass();
        Vector3 initial_base(robot.data()->qpos[0], robot.data()->qpos[1], robot.data()->qpos[2]);
        std::array<Vector3, go2wbc::kNumFeet> initial_feet;
        std::array<Vector3, go2wbc::kNumFeet> locked_feet;
        for (Foot foot : go2wbc::allFeet()) {
            initial_feet[static_cast<size_t>(foot)] = robot.geomPosition(go2wbc::footName(foot));
            locked_feet[static_cast<size_t>(foot)] = initial_feet[static_cast<size_t>(foot)];
        }

        std::vector<TrotWindow> windows = buildWindows(cycles, swing_duration, stance_gap);
        double period = 2.0 * (swing_duration + stance_gap);
        double command_start = windows.empty() ? 0.0 : windows[0].start;

        CentroidalMpcConfig mpc_cfg;
        CentroidalMpc mpc(mpc_cfg);

        GeneralContactWbcConfig stance_cfg;
        stance_cfg.stance_feet = {FOOT_FL, FOOT_FR, FOOT_RL, FOOT_RR};
        stance_cfg.swing_feet = {};
        stance_cfg.normal_force_min = 5.0;
        stance_cfg.weight_force = 1.0;
        stance_cfg.kp_stance = 100.0;
        stance_cfg.kd_stance = 20.0;
        GeneralContactWbc stance_wbc(stance_cfg);

        GeneralContactWbcConfig flrr_cfg;
        flrr_cfg.stance_feet = {FOOT_FR, FOOT_RL};
        flrr_cfg.swing_feet = {FOOT_FL, FOOT_RR};
        flrr_cfg.normal_force_min = 5.0;
        flrr_cfg.weight_swing_foot = 1400.0;
        flrr_cfg.weight_force = 1.0;
        flrr_cfg.weight_base_ori = 300.0;
        flrr_cfg.kp_swing = 450.0;
        flrr_cfg.kd_swing = 42.0;
        flrr_cfg.kp_base_ori = 240.0;
        flrr_cfg.kd_base_ori = 40.0;
        flrr_cfg.kp_stance = 100.0;
        flrr_cfg.kd_stance = 20.0;
        GeneralContactWbc flrr_wbc(flrr_cfg);

        GeneralContactWbcConfig frrl_cfg = flrr_cfg;
        frrl_cfg.stance_feet = {FOOT_FL, FOOT_RR};
        frrl_cfg.swing_feet = {FOOT_FR, FOOT_RL};
        GeneralContactWbc frrl_wbc(frrl_cfg);

        double sim_duration = windows.back().end() + 1.0;
        double mpc_dt = 0.08;
        double wbc_dt = 0.02;
        double next_mpc = 0.0;
        double next_wbc = 0.0;
        double next_log = 0.0;
        double next_record = 0.0;
        double record_dt = 1.0 / 60.0;
        int next_window = 0;
        int active_window = -1;
        std::array<SwingPlan, 2> active_plans;
        VectorX force_ref_all = VectorX::Zero(3 * go2wbc::kNumFeet);
        VectorX tau = VectorX::Zero(robot.nu());
        std::string mpc_status = "not_run";
        std::string wbc_status = "not_run";
        TimerStats stats;
        double wall_start = nowMs();
        std::ofstream record_csv;
        if (!record_csv_path.empty()) {
            std::filesystem::path path(record_csv_path);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            record_csv.open(record_csv_path.c_str(), std::ios::out | std::ios::trunc);
            if (!record_csv.is_open()) {
                throw std::runtime_error("Could not open CSV output: " + record_csv_path);
            }
            writeCsvHeader(record_csv, robot.nq(), robot.nv());
            writeCsvSample(record_csv, robot);
        }

        std::cout << "C++ trot rollout mode=" << (route_mode ? "route" : "constant")
                  << " vx=" << vx
                  << " yaw_rate=" << yaw_rate
                  << " cycles=" << cycles
                  << " command_duration=" << command_duration << "\n";
        if (!record_csv_path.empty()) {
            std::cout << "record_csv=" << record_csv_path << "\n";
        }

        while (robot.data()->time < sim_duration) {
            double sim_time = robot.data()->time;

            if (active_window < 0 && next_window < static_cast<int>(windows.size()) && sim_time >= windows[static_cast<size_t>(next_window)].start) {
                active_window = next_window;
                const TrotWindow& window = windows[static_cast<size_t>(active_window)];
                double elapsed_before = 0.0;
                CommandSegment swing_command = commandAt(command_segments, std::max(0.0, window.start - command_start), &elapsed_before);
                Vector3 nominal_step = limitedPlanarDelta(Vector3(swing_command.vx * period, swing_command.vy * period, 0.0), max_step_length);
                double nominal_yaw_delta = swing_command.yaw_rate * period;
                for (int i = 0; i < 2; ++i) {
                    Foot foot = window.swing_feet[static_cast<size_t>(i)];
                    active_plans[static_cast<size_t>(i)].foot = foot;
                    active_plans[static_cast<size_t>(i)].start_position = locked_feet[static_cast<size_t>(foot)];
                    active_plans[static_cast<size_t>(i)].target_position =
                        locked_feet[static_cast<size_t>(foot)]
                        + footholdDeltaForFoot(foot, initial_feet, nominal_step, nominal_yaw_delta, max_step_length);
                }
                next_mpc = sim_time;
                next_wbc = sim_time;
            }

            TrotWindow* current_window = active_window >= 0 ? &windows[static_cast<size_t>(active_window)] : 0;
            if (current_window != 0 && sim_time >= current_window->end()) {
                for (int i = 0; i < 2; ++i) {
                    Foot foot = active_plans[static_cast<size_t>(i)].foot;
                    locked_feet[static_cast<size_t>(foot)] = active_plans[static_cast<size_t>(i)].target_position;
                }
                active_window = -1;
                next_window++;
                current_window = 0;
                next_mpc = sim_time;
                next_wbc = sim_time;
            }

            std::array<SwingReference, go2wbc::kNumFeet> swing_refs;
            std::array<Vector3, go2wbc::kNumFeet> planned_feet = locked_feet;
            if (current_window != 0) {
                for (int i = 0; i < 2; ++i) {
                    const SwingPlan& plan = active_plans[static_cast<size_t>(i)];
                    SwingReference ref = swingReference(
                        plan.start_position,
                        plan.target_position - plan.start_position,
                        swing_height,
                        current_window->start,
                        current_window->duration,
                        sim_time
                    );
                    swing_refs[static_cast<size_t>(plan.foot)] = ref;
                    planned_feet[static_cast<size_t>(plan.foot)] = ref.position;
                }
            }

            double command_time = std::max(0.0, sim_time - command_start);
            double elapsed_before = 0.0;
            CommandSegment current_command = commandAt(command_segments, command_time, &elapsed_before);
            Vector3 integrated_pose = integratedCommandPose(command_segments, command_time);
            double yaw_ref = integrated_pose(2);
            VectorX qpos_ref = footCenteredBaseReference(home_qpos, initial_base, initial_feet, planned_feet, yaw_ref);
            qpos_ref(1) = initial_base(1) + integrated_pose(1);
            Vector3 com_ref = home_com;
            com_ref(0) += qpos_ref(0) - initial_base(0);
            com_ref(1) += qpos_ref(1) - initial_base(1);
            Vector3 com_vel_ref(current_command.vx, current_command.vy, 0.0);
            Vector3 ori_ref(0.0, 0.0, yaw_ref);
            Vector3 omega_ref(0.0, 0.0, current_command.yaw_rate);

            if (sim_time >= next_mpc) {
                double t0 = nowMs();
                CentroidalMpcInput mpc_input = makeMpcInput(robot, mpc_cfg, windows, current_window, sim_time, com_ref, com_vel_ref, ori_ref, omega_ref);
                go2wbc::CentroidalMpcOutput mpc_out = mpc.solve(robot, mpc_input);
                stats.mpc_ms += nowMs() - t0;
                stats.mpc_count++;
                mpc_status = mpc_out.status;
                if (isSolved(mpc_status)) {
                    force_ref_all = mpc_out.first_contact_forces;
                }
                next_mpc += mpc_dt;
            }

            if (sim_time >= next_wbc) {
                double t0 = nowMs();
                std::vector<Foot> stance_feet = stanceFeetForWindow(current_window);
                GeneralContactWbcInput wbc_input;
                wbc_input.qpos_ref = qpos_ref;
                wbc_input.force_ref = forceRefForFeet(force_ref_all, stance_feet);
                wbc_input.force_zero_weights = VectorX();
                for (Foot foot : stance_feet) {
                    wbc_input.stance_refs[static_cast<size_t>(foot)].enabled = true;
                    wbc_input.stance_refs[static_cast<size_t>(foot)].position = locked_feet[static_cast<size_t>(foot)];
                }
                if (current_window != 0) {
                    for (int i = 0; i < 2; ++i) {
                        Foot foot = active_plans[static_cast<size_t>(i)].foot;
                        wbc_input.swing_refs[static_cast<size_t>(foot)].enabled = true;
                        wbc_input.swing_refs[static_cast<size_t>(foot)].position = swing_refs[static_cast<size_t>(foot)].position;
                        wbc_input.swing_refs[static_cast<size_t>(foot)].velocity = swing_refs[static_cast<size_t>(foot)].velocity;
                        wbc_input.swing_refs[static_cast<size_t>(foot)].acceleration = swing_refs[static_cast<size_t>(foot)].acceleration;
                    }
                }

                GeneralContactWbc* wbc = &stance_wbc;
                if (current_window != 0 && containsFoot(current_window->swing_feet, FOOT_FL)) {
                    wbc = &flrr_wbc;
                } else if (current_window != 0) {
                    wbc = &frrl_wbc;
                }
                go2wbc::GeneralContactWbcOutput wbc_out = wbc->solve(robot, wbc_input);
                stats.wbc_ms += nowMs() - t0;
                stats.wbc_count++;
                wbc_status = wbc_out.status;
                if (isSolved(wbc_status) && isSolved(mpc_status)) {
                    tau = wbc_out.tau;
                }
                next_wbc += wbc_dt;
            }

            for (int i = 0; i < robot.nu(); ++i) {
                robot.data()->ctrl[i] = tau(i);
            }

            double step_t0 = nowMs();
            mj_step(robot.model(), robot.data());
            stats.step_ms += nowMs() - step_t0;
            stats.step_count++;

            if (record_csv.is_open() && robot.data()->time >= next_record) {
                writeCsvSample(record_csv, robot);
                next_record += record_dt;
            }

            if (robot.data()->time >= next_log) {
                Vector3 rpy = go2wbc::quatToRpy(robot.data()->qpos + 3);
                std::cout << "t=" << robot.data()->time
                          << " phase=" << (current_window == 0 ? "stance" : "swing")
                          << " base=[" << robot.data()->qpos[0] << ", " << robot.data()->qpos[1] << ", " << robot.data()->qpos[2] << "]"
                          << " rpy=[" << rpy(0) << ", " << rpy(1) << ", " << rpy(2) << "]"
                          << " tau_max=" << maxAbs(tau)
                          << " mpc=" << mpc_status
                          << " wbc=" << wbc_status << "\n";
                next_log += 1.0;
            }

            Vector3 rpy = go2wbc::quatToRpy(robot.data()->qpos + 3);
            if (robot.data()->qpos[2] < 0.12 || std::abs(rpy(0)) > 0.8 || std::abs(rpy(1)) > 0.8) {
                std::cout << "fall_detected t=" << robot.data()->time
                          << " base_z=" << robot.data()->qpos[2]
                          << " roll=" << rpy(0)
                          << " pitch=" << rpy(1) << "\n";
                break;
            }
        }

        double wall_s = (nowMs() - wall_start) / 1000.0;
        double sim_s = robot.data()->time;
        std::cout << "done sim_time=" << sim_s
                  << " wall_time=" << wall_s
                  << " sim_per_wall=" << (wall_s > 0.0 ? sim_s / wall_s : 0.0) << "\n";
        std::cout << "avg_ms mpc=" << (stats.mpc_count > 0 ? stats.mpc_ms / stats.mpc_count : 0.0)
                  << " wbc=" << (stats.wbc_count > 0 ? stats.wbc_ms / stats.wbc_count : 0.0)
                  << " mj_step=" << (stats.step_count > 0 ? stats.step_ms / stats.step_count : 0.0) << "\n";
        if (record_csv.is_open()) {
            record_csv.close();
            std::cout << "saved_csv=" << record_csv_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
