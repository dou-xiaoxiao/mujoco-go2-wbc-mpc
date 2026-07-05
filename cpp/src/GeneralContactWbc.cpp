#include "go2wbc/GeneralContactWbc.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2wbc {

namespace {

const double kInf = std::numeric_limits<double>::infinity();

Eigen::Quaterniond quatFromWxyz(const double* q) {
    Eigen::Quaterniond quat(q[0], q[1], q[2], q[3]);
    quat.normalize();
    return quat;
}

Vector3 geomPosition(MujocoModelInterface& robot, Foot foot) {
    return robot.geomPosition(footName(foot));
}

Vector3 geomVelocity(MujocoModelInterface& robot, Foot foot) {
    return robot.geomVelocity(footName(foot));
}

FrameJacobian geomJacobian(MujocoModelInterface& robot, Foot foot) {
    return robot.geomJacobian(footName(foot));
}

}  // namespace

GeneralContactWbcConfig::GeneralContactWbcConfig()
    : stance_feet(),
      swing_feet(),
      friction_mu(0.6),
      normal_force_min(0.0),
      weight_base_pos(200.0),
      weight_base_ori(100.0),
      weight_joint_posture(5.0),
      weight_tau(1.0e-4),
      weight_force(1.0),
      weight_swing_foot(200.0),
      kp_base_pos(80.0),
      kd_base_pos(12.0),
      kp_base_ori(120.0),
      kd_base_ori(10.0),
      kp_joint(20.0),
      kd_joint(2.0),
      kp_swing(500.0),
      kd_swing(30.0),
      kp_stance(100.0),
      kd_stance(20.0),
      use_jdot_v(false) {}

FootReference::FootReference()
    : position(Vector3::Zero()),
      velocity(Vector3::Zero()),
      acceleration(Vector3::Zero()),
      enabled(false) {}

GeneralContactWbc::GeneralContactWbc(const GeneralContactWbcConfig& config)
    : config_(config) {
    solver_.setTolerances(1.0e-6, 1.0e-6);
    solver_.setMaxIterations(10000);
    solver_.setPolishing(true);
}

GeneralContactWbcOutput GeneralContactWbc::solve(MujocoModelInterface& robot, const GeneralContactWbcInput& input) {
    QpProblem problem = buildProblem(robot, input);
    QpSolution qp = solver_.solve(problem);

    const int nv = robot.nv();
    const int nu = robot.nu();
    const int nf = 3 * static_cast<int>(config_.stance_feet.size());

    GeneralContactWbcOutput out;
    out.vdot = qp.x.segment(0, nv);
    out.tau = qp.x.segment(nv, nu);
    out.contact_forces = qp.x.segment(nv + nu, nf);
    out.status = qp.status;
    out.status_value = qp.status_value;
    out.objective = qp.objective;
    out.iterations = qp.iterations;

    MatrixX mass = robot.massMatrix();
    VectorX h = robot.biasForces(false);
    MatrixX B = robot.actuationMatrix();
    MatrixX Jc = stackedJacobian(robot, config_.stance_feet);
    VectorX stance_cmd = stanceAccelCmd(robot, input);

    out.dynamics_residual = mass * out.vdot + h - B * out.tau - Jc.transpose() * out.contact_forces;
    out.stance_residual = Jc * out.vdot - stance_cmd;

    out.swing_accel_error = VectorX::Zero(3 * static_cast<int>(config_.swing_feet.size()));
    for (int i = 0; i < static_cast<int>(config_.swing_feet.size()); ++i) {
        Foot foot = config_.swing_feet[static_cast<size_t>(i)];
        FrameJacobian jac = geomJacobian(robot, foot);
        Vector3 current_pos = geomPosition(robot, foot);
        Vector3 current_vel = jac.jacp * robot.qvel();
        FootReference ref = input.swing_refs[static_cast<size_t>(foot)];
        Vector3 pos_ref = ref.enabled ? ref.position : current_pos;
        Vector3 vel_ref = ref.enabled ? ref.velocity : Vector3::Zero();
        Vector3 acc_ref = ref.enabled ? ref.acceleration : Vector3::Zero();
        Vector3 cmd = acc_ref
            + config_.kp_swing * (pos_ref - current_pos)
            + config_.kd_swing * (vel_ref - current_vel);
        out.swing_accel_error.segment(3 * i, 3) = jac.jacp * out.vdot - cmd;
    }

    return out;
}

QpProblem GeneralContactWbc::buildProblem(MujocoModelInterface& robot, const GeneralContactWbcInput& input) {
    if (config_.stance_feet.empty()) {
        throw std::runtime_error("GeneralContactWbc requires at least one stance foot.");
    }
    const int nv = robot.nv();
    const int nu = robot.nu();
    const int nf = 3 * static_cast<int>(config_.stance_feet.size());
    const int nvar = nv + nu + nf;

    MatrixX M = robot.massMatrix();
    VectorX h = robot.biasForces(false);
    MatrixX B = robot.actuationMatrix();
    MatrixX Jc = stackedJacobian(robot, config_.stance_feet);
    VectorX stance_cmd = stanceAccelCmd(robot, input);
    Vector3 base_pos_cmd = basePositionAccelCmd(robot, input.qpos_ref);
    Vector3 base_ori_cmd = baseOrientationAccelCmd(robot, input.qpos_ref);
    VectorX joint_cmd = jointAccelCmd(robot, input.qpos_ref);

    VectorX force_ref = input.force_ref;
    if (force_ref.size() != nf) {
        force_ref = defaultForceReference(robot, static_cast<int>(config_.stance_feet.size()));
    }
    VectorX force_zero_weights = input.force_zero_weights;
    if (force_zero_weights.size() != nf) {
        force_zero_weights = VectorX::Zero(nf);
    }

    MatrixX P = MatrixX::Zero(nvar, nvar);
    VectorX q = VectorX::Zero(nvar);

    addDiagonalTrackingCost(P, q, 0, 3, config_.weight_base_pos, base_pos_cmd);
    addDiagonalTrackingCost(P, q, 3, 3, config_.weight_base_ori, base_ori_cmd);
    addDiagonalTrackingCost(P, q, 6, nv - 6, config_.weight_joint_posture, joint_cmd);

    for (int i = 0; i < nu; ++i) {
        P(nv + i, nv + i) += config_.weight_tau;
    }
    for (int i = 0; i < nf; ++i) {
        int index = nv + nu + i;
        P(index, index) += config_.weight_force + force_zero_weights(i);
        q(index) += -config_.weight_force * force_ref(i);
    }

    for (size_t swing_id = 0; swing_id < config_.swing_feet.size(); ++swing_id) {
        Foot foot = config_.swing_feet[swing_id];
        FrameJacobian jac = geomJacobian(robot, foot);
        Vector3 current_pos = geomPosition(robot, foot);
        Vector3 current_vel = jac.jacp * robot.qvel();
        FootReference ref = input.swing_refs[static_cast<size_t>(foot)];
        Vector3 pos_ref = ref.enabled ? ref.position : current_pos;
        Vector3 vel_ref = ref.enabled ? ref.velocity : Vector3::Zero();
        Vector3 acc_ref = ref.enabled ? ref.acceleration : Vector3::Zero();
        Vector3 cmd = acc_ref
            + config_.kp_swing * (pos_ref - current_pos)
            + config_.kd_swing * (vel_ref - current_vel);

        P.block(0, 0, nv, nv) += config_.weight_swing_foot * (jac.jacp.transpose() * jac.jacp);
        q.segment(0, nv) += -config_.weight_swing_foot * jac.jacp.transpose() * cmd;
    }

    for (int i = 0; i < nvar; ++i) {
        P(i, i) += 1.0e-9;
    }

    const int dyn_rows = nv;
    const int stance_rows = nf;
    const int friction_rows = 5 * static_cast<int>(config_.stance_feet.size());
    const int torque_rows = nu;
    const int ncon = dyn_rows + stance_rows + friction_rows + torque_rows;

    MatrixX A = MatrixX::Zero(ncon, nvar);
    VectorX lower = VectorX::Zero(ncon);
    VectorX upper = VectorX::Zero(ncon);
    int row = 0;

    A.block(row, 0, nv, nv) = M;
    A.block(row, nv, nv, nu) = -B;
    A.block(row, nv + nu, nv, nf) = -Jc.transpose();
    lower.segment(row, nv) = -h;
    upper.segment(row, nv) = -h;
    row += nv;

    A.block(row, 0, nf, nv) = Jc;
    lower.segment(row, nf) = stance_cmd;
    upper.segment(row, nf) = stance_cmd;
    row += nf;

    for (int contact = 0; contact < static_cast<int>(config_.stance_feet.size()); ++contact) {
        int fx = nv + nu + 3 * contact;
        int fy = fx + 1;
        int fz = fx + 2;

        A(row, fx) = 1.0;
        A(row, fz) = -config_.friction_mu;
        lower(row) = -kInf;
        upper(row) = 0.0;
        row++;

        A(row, fx) = -1.0;
        A(row, fz) = -config_.friction_mu;
        lower(row) = -kInf;
        upper(row) = 0.0;
        row++;

        A(row, fy) = 1.0;
        A(row, fz) = -config_.friction_mu;
        lower(row) = -kInf;
        upper(row) = 0.0;
        row++;

        A(row, fy) = -1.0;
        A(row, fz) = -config_.friction_mu;
        lower(row) = -kInf;
        upper(row) = 0.0;
        row++;

        A(row, fz) = -1.0;
        lower(row) = -kInf;
        upper(row) = -config_.normal_force_min;
        row++;
    }

    const mjModel* model = robot.model();
    for (int actuator = 0; actuator < nu; ++actuator) {
        A(row, nv + actuator) = 1.0;
        lower(row) = model->actuator_ctrlrange[2 * actuator + 0];
        upper(row) = model->actuator_ctrlrange[2 * actuator + 1];
        row++;
    }

    QpProblem problem;
    problem.P = denseToCSC(P, true);
    problem.q = q;
    problem.A = denseToCSC(A, false);
    problem.lower = lower;
    problem.upper = upper;
    return problem;
}

Vector3 GeneralContactWbc::basePositionAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const {
    Vector3 err;
    err << qpos_ref(0) - robot.data()->qpos[0],
           qpos_ref(1) - robot.data()->qpos[1],
           qpos_ref(2) - robot.data()->qpos[2];
    Vector3 vel(robot.data()->qvel[0], robot.data()->qvel[1], robot.data()->qvel[2]);
    return config_.kp_base_pos * err - config_.kd_base_pos * vel;
}

Vector3 GeneralContactWbc::baseOrientationAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const {
    double desired[4] = {qpos_ref(3), qpos_ref(4), qpos_ref(5), qpos_ref(6)};
    Vector3 err = quatErrorRotvec(desired, robot.data()->qpos + 3);
    Vector3 omega(robot.data()->qvel[3], robot.data()->qvel[4], robot.data()->qvel[5]);
    return config_.kp_base_ori * err - config_.kd_base_ori * omega;
}

VectorX GeneralContactWbc::jointAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const {
    VectorX cmd(robot.nu());
    for (int i = 0; i < robot.nu(); ++i) {
        double qerr = qpos_ref(7 + i) - robot.data()->qpos[7 + i];
        double qd = robot.data()->qvel[6 + i];
        cmd(i) = config_.kp_joint * qerr - config_.kd_joint * qd;
    }
    return cmd;
}

VectorX GeneralContactWbc::stanceAccelCmd(MujocoModelInterface& robot, const GeneralContactWbcInput& input) const {
    VectorX cmd = VectorX::Zero(3 * static_cast<int>(config_.stance_feet.size()));
    if (config_.kp_stance == 0.0 && config_.kd_stance == 0.0) {
        return cmd;
    }
    for (int i = 0; i < static_cast<int>(config_.stance_feet.size()); ++i) {
        Foot foot = config_.stance_feet[static_cast<size_t>(i)];
        const FootReference& ref = input.stance_refs[static_cast<size_t>(foot)];
        if (!ref.enabled) {
            continue;
        }
        Vector3 pos_err = ref.position - geomPosition(robot, foot);
        Vector3 vel_err = -geomVelocity(robot, foot);
        cmd.segment(3 * i, 3) = config_.kp_stance * pos_err + config_.kd_stance * vel_err;
    }
    return cmd;
}

VectorX GeneralContactWbc::defaultForceReference(MujocoModelInterface& robot, int num_contacts) const {
    VectorX ref = VectorX::Zero(3 * num_contacts);
    double fz = robot.totalMass() * std::abs(robot.model()->opt.gravity[2]) / static_cast<double>(num_contacts);
    for (int i = 0; i < num_contacts; ++i) {
        ref(3 * i + 2) = fz;
    }
    return ref;
}

MatrixX GeneralContactWbc::stackedJacobian(MujocoModelInterface& robot, const std::vector<Foot>& feet) const {
    MatrixX J(3 * static_cast<int>(feet.size()), robot.nv());
    for (int i = 0; i < static_cast<int>(feet.size()); ++i) {
        FrameJacobian jac = geomJacobian(robot, feet[static_cast<size_t>(i)]);
        J.block(3 * i, 0, 3, robot.nv()) = jac.jacp;
    }
    return J;
}

void GeneralContactWbc::addDiagonalTrackingCost(
    MatrixX& P,
    VectorX& q,
    int start,
    int count,
    double weight,
    const VectorX& target
) const {
    for (int i = 0; i < count; ++i) {
        P(start + i, start + i) += weight;
        q(start + i) += -weight * target(i);
    }
}

Vector3 quatErrorRotvec(const double* desired_wxyz, const double* current_wxyz) {
    Eigen::Quaterniond desired = quatFromWxyz(desired_wxyz);
    Eigen::Quaterniond current = quatFromWxyz(current_wxyz);
    Eigen::Quaterniond error = desired * current.conjugate();
    if (error.w() < 0.0) {
        error.coeffs() *= -1.0;
    }
    return Vector3(2.0 * error.x(), 2.0 * error.y(), 2.0 * error.z());
}

}  // namespace go2wbc
