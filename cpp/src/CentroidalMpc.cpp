#include "go2wbc/CentroidalMpc.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace go2wbc {

namespace {

const int kStateDim = 12;
const int kForceDimAll = 3 * kNumFeet;
const double kInf = std::numeric_limits<double>::infinity();

int stateIndex(int step, int offset) {
    return step * kStateDim + offset;
}

int forceIndex(int force_offset, int step, int offset) {
    return force_offset + step * kForceDimAll + offset;
}

struct SparseEntry {
    int row;
    int col;
    double value;
};

void addEntry(std::vector<SparseEntry>* entries, int row, int col, double value) {
    SparseEntry entry;
    entry.row = row;
    entry.col = col;
    entry.value = value;
    entries->push_back(entry);
}

SparseCSC entriesToCSC(int rows, int cols, std::vector<SparseEntry> entries) {
    std::sort(entries.begin(), entries.end(), [](const SparseEntry& a, const SparseEntry& b) {
        if (a.col != b.col) {
            return a.col < b.col;
        }
        return a.row < b.row;
    });

    SparseCSC out;
    out.rows = rows;
    out.cols = cols;
    out.col_ptr.assign(static_cast<size_t>(cols + 1), 0);

    size_t cursor = 0;
    for (int col = 0; col < cols; ++col) {
        out.col_ptr[static_cast<size_t>(col)] = static_cast<OSQPInt>(out.values.size());
        while (cursor < entries.size() && entries[cursor].col == col) {
            int row = entries[cursor].row;
            double value = 0.0;
            while (cursor < entries.size() && entries[cursor].col == col && entries[cursor].row == row) {
                value += entries[cursor].value;
                cursor++;
            }
            out.row_idx.push_back(static_cast<OSQPInt>(row));
            out.values.push_back(static_cast<OSQPFloat>(value));
        }
    }
    out.col_ptr[static_cast<size_t>(cols)] = static_cast<OSQPInt>(out.values.size());
    return out;
}

}  // namespace

CentroidalMpcConfig::CentroidalMpcConfig()
    : horizon_steps(12),
      dt(0.03),
      friction_mu(0.6),
      normal_force_min(5.0),
      weight_com_position(500.0),
      weight_com_velocity(20.0),
      weight_orientation(1200.0),
      weight_angular_velocity(100.0),
      weight_force_regularization(1.0e-4),
      weight_force_rate(1.0e-5) {}

CentroidalMpc::CentroidalMpc(const CentroidalMpcConfig& config)
    : config_(config) {
    solver_.setTolerances(1.0e-6, 1.0e-6);
    solver_.setMaxIterations(4000);
    solver_.setPolishing(false);
}

CentroidalMpcOutput CentroidalMpc::solve(MujocoModelInterface& robot, const CentroidalMpcInput& input) {
    MatrixX torque_map;
    QpProblem problem = buildProblem(robot, input, &torque_map);
    QpSolution qp = solver_.solve(problem);

    int n_steps = config_.horizon_steps;
    int n_state_vars = (n_steps + 1) * kStateDim;

    CentroidalMpcOutput out;
    out.states = MatrixX::Zero(n_steps + 1, kStateDim);
    for (int step = 0; step <= n_steps; ++step) {
        out.states.row(step) = qp.x.segment(step * kStateDim, kStateDim).transpose();
    }

    out.contact_forces.resize(static_cast<size_t>(n_steps));
    for (int step = 0; step < n_steps; ++step) {
        for (Foot foot : allFeet()) {
            int foot_id = static_cast<int>(foot);
            int base = n_state_vars + step * kForceDimAll + 3 * foot_id;
            out.contact_forces[static_cast<size_t>(step)][static_cast<size_t>(foot)] =
                qp.x.segment(base, 3);
        }
    }

    out.first_contact_forces = qp.x.segment(n_state_vars, kForceDimAll);
    out.dynamics_residual = computeDynamicsResidual(out.states, out.contact_forces, robot, torque_map);
    out.status = qp.status;
    out.status_value = qp.status_value;
    out.objective = qp.objective;
    out.iterations = qp.iterations;
    return out;
}

QpProblem CentroidalMpc::buildProblem(MujocoModelInterface& robot, const CentroidalMpcInput& input, MatrixX* torque_map_out) {
    int n_steps = config_.horizon_steps;
    int n_state_vars = (n_steps + 1) * kStateDim;
    int n_force_vars = n_steps * kForceDimAll;
    int nvar = n_state_vars + n_force_vars;

    if (static_cast<int>(input.contact_schedule.size()) != n_steps) {
        throw std::runtime_error("CentroidalMpc contact_schedule has the wrong horizon length.");
    }

    MatrixX com_pos_ref = expandReference(input.com_position_ref, n_steps + 1);
    MatrixX com_vel_ref = expandReference(input.com_velocity_ref, n_steps + 1);
    MatrixX ori_ref = expandReference(input.orientation_ref, n_steps + 1);
    MatrixX omega_ref = expandReference(input.angular_velocity_ref, n_steps + 1);

    double mass = robot.totalMass();
    Vector3 gravity(robot.model()->opt.gravity[0], robot.model()->opt.gravity[1], robot.model()->opt.gravity[2]);
    Vector3 com = robot.centerOfMass();

    std::array<Vector3, kNumFeet> contact_positions;
    for (Foot foot : allFeet()) {
        contact_positions[static_cast<size_t>(foot)] = robot.geomPosition(footName(foot));
    }

    MatrixX inertia_inv = robot.compositeInertiaWorldAboutCom().inverse();
    MatrixX angular_map = torqueMap(contact_positions, com, inertia_inv);
    if (torque_map_out != 0) {
        *torque_map_out = angular_map;
    }

    VectorX x0(kStateDim);
    x0.segment(0, 3) = com;
    x0.segment(3, 3) = robot.baseLinearVelocity();
    x0.segment(6, 3) = quatToRpy(robot.data()->qpos + 3);
    x0.segment(9, 3) = robot.baseAngularVelocity();

    std::vector<SparseEntry> p_entries;
    std::vector<SparseEntry> a_entries;
    p_entries.reserve(static_cast<size_t>(nvar + n_steps * kForceDimAll));
    a_entries.reserve(static_cast<size_t>(12 + n_steps * (3 * 3 + 3 * 6 + 3 * 3 + 3 * 14 + kNumFeet * 9)));
    VectorX q = VectorX::Zero(nvar);

    std::vector<double> p_diag(static_cast<size_t>(nvar), 1.0e-9);

    for (int step = 1; step <= n_steps; ++step) {
        int base = step * kStateDim;
        for (int axis = 0; axis < 3; ++axis) {
            p_diag[static_cast<size_t>(base + axis)] += config_.weight_com_position;
            q(base + axis) += -config_.weight_com_position * com_pos_ref(step, axis);
            p_diag[static_cast<size_t>(base + 3 + axis)] += config_.weight_com_velocity;
            q(base + 3 + axis) += -config_.weight_com_velocity * com_vel_ref(step, axis);
            p_diag[static_cast<size_t>(base + 6 + axis)] += config_.weight_orientation;
            q(base + 6 + axis) += -config_.weight_orientation * ori_ref(step, axis);
            p_diag[static_cast<size_t>(base + 9 + axis)] += config_.weight_angular_velocity;
            q(base + 9 + axis) += -config_.weight_angular_velocity * omega_ref(step, axis);
        }
    }

    for (int step = 0; step < n_steps; ++step) {
        int stance_count = 0;
        for (Foot foot : allFeet()) {
            if (input.contact_schedule[static_cast<size_t>(step)][static_cast<size_t>(foot)]) {
                stance_count++;
            }
        }
        if (stance_count <= 0) {
            stance_count = 1;
        }
        double fz_ref = mass * std::abs(gravity(2)) / static_cast<double>(stance_count);
        for (Foot foot : allFeet()) {
            int foot_id = static_cast<int>(foot);
            int base = forceIndex(n_state_vars, step, 3 * foot_id);
            for (int axis = 0; axis < 3; ++axis) {
                p_diag[static_cast<size_t>(base + axis)] += config_.weight_force_regularization;
            }
            if (input.contact_schedule[static_cast<size_t>(step)][static_cast<size_t>(foot)]) {
                q(base + 2) += -config_.weight_force_regularization * fz_ref;
            }
        }
    }

    if (config_.weight_force_rate > 0.0) {
        for (int step = 1; step < n_steps; ++step) {
            int prev = forceIndex(n_state_vars, step - 1, 0);
            int curr = forceIndex(n_state_vars, step, 0);
            for (int idx = 0; idx < kForceDimAll; ++idx) {
                p_diag[static_cast<size_t>(prev + idx)] += config_.weight_force_rate;
                p_diag[static_cast<size_t>(curr + idx)] += config_.weight_force_rate;
                addEntry(&p_entries, prev + idx, curr + idx, -config_.weight_force_rate);
            }
        }
    }

    for (int i = 0; i < nvar; ++i) {
        addEntry(&p_entries, i, i, p_diag[static_cast<size_t>(i)]);
    }

    int dyn_rows = (n_steps + 1) * kStateDim;
    int force_rows = n_steps * kNumFeet * 5;
    int ncon = dyn_rows + force_rows;
    MatrixX A = MatrixX::Zero(ncon, nvar);
    VectorX lower = VectorX::Zero(ncon);
    VectorX upper = VectorX::Zero(ncon);
    int row = 0;

    for (int idx = 0; idx < kStateDim; ++idx) {
        addEntry(&a_entries, row, idx, 1.0);
        lower(row) = x0(idx);
        upper(row) = x0(idx);
        row++;
    }

    for (int step = 0; step < n_steps; ++step) {
        int xk = step * kStateDim;
        int xkp1 = (step + 1) * kStateDim;
        int uk = n_state_vars + step * kForceDimAll;

        for (int axis = 0; axis < 3; ++axis) {
            addEntry(&a_entries, row, xkp1 + axis, 1.0);
            addEntry(&a_entries, row, xk + axis, -1.0);
            addEntry(&a_entries, row, xk + 3 + axis, -config_.dt);
            lower(row) = 0.0;
            upper(row) = 0.0;
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            addEntry(&a_entries, row, xkp1 + 3 + axis, 1.0);
            addEntry(&a_entries, row, xk + 3 + axis, -1.0);
            for (int force_axis = axis; force_axis < kForceDimAll; force_axis += 3) {
                addEntry(&a_entries, row, uk + force_axis, -config_.dt / mass);
            }
            lower(row) = config_.dt * gravity(axis);
            upper(row) = config_.dt * gravity(axis);
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            addEntry(&a_entries, row, xkp1 + 6 + axis, 1.0);
            addEntry(&a_entries, row, xk + 6 + axis, -1.0);
            addEntry(&a_entries, row, xk + 9 + axis, -config_.dt);
            lower(row) = 0.0;
            upper(row) = 0.0;
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            addEntry(&a_entries, row, xkp1 + 9 + axis, 1.0);
            addEntry(&a_entries, row, xk + 9 + axis, -1.0);
            for (int force_idx = 0; force_idx < kForceDimAll; ++force_idx) {
                addEntry(&a_entries, row, uk + force_idx, -config_.dt * angular_map(axis, force_idx));
            }
            lower(row) = 0.0;
            upper(row) = 0.0;
            row++;
        }
    }

    for (int step = 0; step < n_steps; ++step) {
        for (Foot foot : allFeet()) {
            int foot_id = static_cast<int>(foot);
            bool stance = input.contact_schedule[static_cast<size_t>(step)][static_cast<size_t>(foot)];
            int fx = forceIndex(n_state_vars, step, 3 * foot_id);
            int fy = fx + 1;
            int fz = fx + 2;

            addEntry(&a_entries, row, fx, 1.0);
            addEntry(&a_entries, row, fz, -config_.friction_mu);
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            addEntry(&a_entries, row, fx, -1.0);
            addEntry(&a_entries, row, fz, -config_.friction_mu);
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            addEntry(&a_entries, row, fy, 1.0);
            addEntry(&a_entries, row, fz, -config_.friction_mu);
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            addEntry(&a_entries, row, fy, -1.0);
            addEntry(&a_entries, row, fz, -config_.friction_mu);
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            addEntry(&a_entries, row, fz, 1.0);
            lower(row) = stance ? config_.normal_force_min : 0.0;
            upper(row) = stance ? kInf : 0.0;
            row++;
        }
    }

    QpProblem problem;
    problem.P = entriesToCSC(nvar, nvar, p_entries);
    problem.q = q;
    problem.A = entriesToCSC(ncon, nvar, a_entries);
    problem.lower = lower;
    problem.upper = upper;
    return problem;
}

MatrixX CentroidalMpc::expandReference(const MatrixX& ref, int rows) const {
    if (ref.rows() == rows && ref.cols() == 3) {
        return ref;
    }
    if (ref.rows() == 1 && ref.cols() == 3) {
        MatrixX out(rows, 3);
        for (int row = 0; row < rows; ++row) {
            out.row(row) = ref.row(0);
        }
        return out;
    }
    if (ref.size() == 0) {
        return MatrixX::Zero(rows, 3);
    }
    throw std::runtime_error("CentroidalMpc reference must be empty, 1x3, or (N+1)x3.");
}

MatrixX CentroidalMpc::torqueMap(
    const std::array<Vector3, kNumFeet>& contact_positions,
    const Vector3& com,
    const MatrixX& inertia_inv
) const {
    MatrixX map = MatrixX::Zero(3, kForceDimAll);
    for (Foot foot : allFeet()) {
        int id = static_cast<int>(foot);
        map.block(0, 3 * id, 3, 3) = skewMatrix(contact_positions[static_cast<size_t>(foot)] - com);
    }
    return inertia_inv * map;
}

VectorX CentroidalMpc::computeDynamicsResidual(
    const MatrixX& states,
    const std::vector<std::array<Vector3, kNumFeet> >& forces,
    MujocoModelInterface& robot,
    const MatrixX& torque_map
) const {
    int n_steps = config_.horizon_steps;
    Vector3 gravity(robot.model()->opt.gravity[0], robot.model()->opt.gravity[1], robot.model()->opt.gravity[2]);
    double mass = robot.totalMass();
    VectorX residual = VectorX::Zero(n_steps * kStateDim);
    int out = 0;
    for (int step = 0; step < n_steps; ++step) {
        VectorX force_all(kForceDimAll);
        Vector3 sum_force = Vector3::Zero();
        for (Foot foot : allFeet()) {
            int id = static_cast<int>(foot);
            Vector3 f = forces[static_cast<size_t>(step)][static_cast<size_t>(foot)];
            force_all.segment(3 * id, 3) = f;
            sum_force += f;
        }

        Vector3 pos_next = states.row(step).segment(0, 3).transpose()
            + config_.dt * states.row(step).segment(3, 3).transpose();
        Vector3 vel_next = states.row(step).segment(3, 3).transpose()
            + config_.dt * (sum_force / mass + gravity);
        Vector3 theta_next = states.row(step).segment(6, 3).transpose()
            + config_.dt * states.row(step).segment(9, 3).transpose();
        Vector3 omega_next = states.row(step).segment(9, 3).transpose()
            + config_.dt * (torque_map * force_all);

        residual.segment(out, 3) = states.row(step + 1).segment(0, 3).transpose() - pos_next;
        out += 3;
        residual.segment(out, 3) = states.row(step + 1).segment(3, 3).transpose() - vel_next;
        out += 3;
        residual.segment(out, 3) = states.row(step + 1).segment(6, 3).transpose() - theta_next;
        out += 3;
        residual.segment(out, 3) = states.row(step + 1).segment(9, 3).transpose() - omega_next;
        out += 3;
    }
    return residual;
}

MatrixX skewMatrix(const Vector3& v) {
    MatrixX out(3, 3);
    out << 0.0, -v(2), v(1),
           v(2), 0.0, -v(0),
           -v(1), v(0), 0.0;
    return out;
}

Vector3 quatToRpy(const double* quat_wxyz) {
    double w = quat_wxyz[0];
    double x = quat_wxyz[1];
    double y = quat_wxyz[2];
    double z = quat_wxyz[3];
    double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    double sin_pitch = 2.0 * (w * y - z * x);
    if (sin_pitch > 1.0) {
        sin_pitch = 1.0;
    }
    if (sin_pitch < -1.0) {
        sin_pitch = -1.0;
    }
    double pitch = std::asin(sin_pitch);
    double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    return Vector3(roll, pitch, yaw);
}

}  // namespace go2wbc
