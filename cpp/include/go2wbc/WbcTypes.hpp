#pragma once

#include "go2wbc/Types.hpp"

namespace go2wbc {

struct WbcInput {
    RobotState state;
    ContactState contact;
    VectorX qpos_ref;
    VectorX force_ref_all;
};

struct WbcOutput {
    VectorX vdot;
    VectorX tau;
    VectorX contact_force;
    VectorX dynamics_residual;
    int status;

    WbcOutput() : status(0) {}
};

struct MpcInput {
    RobotState state;
    ContactState contact;
    Vector3 com_position_ref;
    Vector3 com_velocity_ref;
    Vector3 orientation_rpy_ref;
    Vector3 angular_velocity_ref;
};

struct MpcOutput {
    VectorX force_ref_all;
    VectorX dynamics_residual;
    int status;

    MpcOutput() : status(0) {}
};

}  // namespace go2wbc
