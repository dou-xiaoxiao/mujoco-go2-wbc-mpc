#pragma once

#include <array>
#include <vector>

#include "go2wbc/MujocoModelInterface.hpp"
#include "go2wbc/OsqpSolver.hpp"
#include "go2wbc/Types.hpp"

namespace go2wbc {

struct CentroidalMpcConfig {
    int horizon_steps;
    double dt;
    double friction_mu;
    double normal_force_min;
    double weight_com_position;
    double weight_com_velocity;
    double weight_orientation;
    double weight_angular_velocity;
    double weight_force_regularization;
    double weight_force_rate;

    CentroidalMpcConfig();
};

struct CentroidalMpcInput {
    MatrixX com_position_ref;
    MatrixX com_velocity_ref;
    MatrixX orientation_ref;
    MatrixX angular_velocity_ref;
    std::vector<std::array<bool, kNumFeet> > contact_schedule;
};

struct CentroidalMpcOutput {
    MatrixX states;
    std::vector<std::array<Vector3, kNumFeet> > contact_forces;
    VectorX first_contact_forces;
    VectorX dynamics_residual;
    std::string status;
    int status_value;
    double objective;
    int iterations;
};

class CentroidalMpc {
public:
    explicit CentroidalMpc(const CentroidalMpcConfig& config);

    CentroidalMpcOutput solve(MujocoModelInterface& robot, const CentroidalMpcInput& input);

private:
    MatrixX expandReference(const MatrixX& ref, int rows) const;
    MatrixX torqueMap(const std::array<Vector3, kNumFeet>& contact_positions, const Vector3& com, const MatrixX& inertia_inv) const;
    QpProblem buildProblem(MujocoModelInterface& robot, const CentroidalMpcInput& input, MatrixX* torque_map_out);
    VectorX computeDynamicsResidual(
        const MatrixX& states,
        const std::vector<std::array<Vector3, kNumFeet> >& forces,
        MujocoModelInterface& robot,
        const MatrixX& torque_map
    ) const;

    CentroidalMpcConfig config_;
    OsqpSolver solver_;
};

MatrixX skewMatrix(const Vector3& v);
Vector3 quatToRpy(const double* quat_wxyz);

}  // namespace go2wbc
