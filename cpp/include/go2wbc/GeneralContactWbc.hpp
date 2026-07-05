#pragma once

#include <array>
#include <vector>

#include "go2wbc/MujocoModelInterface.hpp"
#include "go2wbc/OsqpSolver.hpp"
#include "go2wbc/Types.hpp"

namespace go2wbc {

struct GeneralContactWbcConfig {
    std::vector<Foot> stance_feet;
    std::vector<Foot> swing_feet;

    double friction_mu;
    double normal_force_min;

    double weight_base_pos;
    double weight_base_ori;
    double weight_joint_posture;
    double weight_tau;
    double weight_force;
    double weight_swing_foot;

    double kp_base_pos;
    double kd_base_pos;
    double kp_base_ori;
    double kd_base_ori;
    double kp_joint;
    double kd_joint;
    double kp_swing;
    double kd_swing;
    double kp_stance;
    double kd_stance;

    bool use_jdot_v;

    GeneralContactWbcConfig();
};

struct FootReference {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    bool enabled;

    FootReference();
};

struct GeneralContactWbcInput {
    VectorX qpos_ref;
    std::array<FootReference, kNumFeet> swing_refs;
    std::array<FootReference, kNumFeet> stance_refs;
    VectorX force_ref;
    VectorX force_zero_weights;
};

struct GeneralContactWbcOutput {
    VectorX vdot;
    VectorX tau;
    VectorX contact_forces;
    VectorX dynamics_residual;
    VectorX stance_residual;
    VectorX swing_accel_error;
    std::string status;
    int status_value;
    double objective;
    int iterations;
};

class GeneralContactWbc {
public:
    explicit GeneralContactWbc(const GeneralContactWbcConfig& config);

    GeneralContactWbcOutput solve(MujocoModelInterface& robot, const GeneralContactWbcInput& input);

private:
    Vector3 basePositionAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const;
    Vector3 baseOrientationAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const;
    VectorX jointAccelCmd(MujocoModelInterface& robot, const VectorX& qpos_ref) const;
    VectorX stanceAccelCmd(MujocoModelInterface& robot, const GeneralContactWbcInput& input) const;
    VectorX defaultForceReference(MujocoModelInterface& robot, int num_contacts) const;
    MatrixX stackedJacobian(MujocoModelInterface& robot, const std::vector<Foot>& feet) const;
    void addDiagonalTrackingCost(MatrixX& P, VectorX& q, int start, int count, double weight, const VectorX& target) const;
    QpProblem buildProblem(MujocoModelInterface& robot, const GeneralContactWbcInput& input);

    GeneralContactWbcConfig config_;
    OsqpSolver solver_;
};

Vector3 quatErrorRotvec(const double* desired_wxyz, const double* current_wxyz);

}  // namespace go2wbc
