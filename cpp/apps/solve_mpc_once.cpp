#include <iostream>
#include <string>

#include "go2wbc/CentroidalMpc.hpp"

using go2wbc::CentroidalMpc;
using go2wbc::CentroidalMpcConfig;
using go2wbc::CentroidalMpcInput;
using go2wbc::FOOT_FL;
using go2wbc::FOOT_RR;
using go2wbc::MatrixX;
using go2wbc::MujocoModelInterface;
using go2wbc::Vector3;
using go2wbc::VectorX;

double maxAbs(const VectorX& v) {
    if (v.size() == 0) {
        return 0.0;
    }
    return v.cwiseAbs().maxCoeff();
}

CentroidalMpcInput makeInput(MujocoModelInterface& robot, const CentroidalMpcConfig& cfg, bool diagonal_swing) {
    CentroidalMpcInput input;
    Vector3 com = robot.centerOfMass();
    input.com_position_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.com_velocity_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.orientation_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    input.angular_velocity_ref = MatrixX::Zero(cfg.horizon_steps + 1, 3);
    for (int k = 0; k <= cfg.horizon_steps; ++k) {
        input.com_position_ref.row(k) = com.transpose();
    }
    input.contact_schedule.resize(static_cast<size_t>(cfg.horizon_steps));
    for (int k = 0; k < cfg.horizon_steps; ++k) {
        input.contact_schedule[static_cast<size_t>(k)].fill(true);
        if (diagonal_swing) {
            input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(FOOT_FL)] = false;
            input.contact_schedule[static_cast<size_t>(k)][static_cast<size_t>(FOOT_RR)] = false;
        }
    }
    return input;
}

void printSolve(const char* label, CentroidalMpc& mpc, MujocoModelInterface& robot, const CentroidalMpcInput& input) {
    go2wbc::CentroidalMpcOutput out = mpc.solve(robot, input);
    std::cout << label << " status=" << out.status
              << " iter=" << out.iterations
              << " obj=" << out.objective
              << " dyn_res=" << maxAbs(out.dynamics_residual) << "\n";
    std::cout << label << " f0=" << out.first_contact_forces.transpose() << "\n";
}

int main(int argc, char** argv) {
    try {
        std::string model_path = "../models/mujoco_menagerie/unitree_go2/scene.xml";
        if (argc >= 2) {
            model_path = argv[1];
        }

        MujocoModelInterface robot(model_path);
        robot.setKeyframe("home");

        CentroidalMpcConfig cfg;
        cfg.horizon_steps = 12;
        cfg.dt = 0.03;
        cfg.normal_force_min = 5.0;
        cfg.weight_orientation = 1200.0;
        cfg.weight_angular_velocity = 100.0;

        CentroidalMpc mpc(cfg);
        printSolve("all_stance", mpc, robot, makeInput(robot, cfg, false));
        printSolve("FL_RR_swing", mpc, robot, makeInput(robot, cfg, true));
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
