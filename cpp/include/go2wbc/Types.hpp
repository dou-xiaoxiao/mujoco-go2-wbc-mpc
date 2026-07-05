#pragma once

#include <array>
#include <string>

#include <Eigen/Dense>

namespace go2wbc {

static const int kNumFeet = 4;
static const int kFootForceDim = 3;
static const int kGo2Nq = 19;
static const int kGo2Nv = 18;
static const int kGo2Nu = 12;

enum Foot {
    FOOT_FL = 0,
    FOOT_FR = 1,
    FOOT_RL = 2,
    FOOT_RR = 3
};

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixX;
typedef Eigen::VectorXd VectorX;
typedef Eigen::Vector3d Vector3;

inline const char* footName(Foot foot) {
    switch (foot) {
        case FOOT_FL: return "FL";
        case FOOT_FR: return "FR";
        case FOOT_RL: return "RL";
        case FOOT_RR: return "RR";
        default: return "";
    }
}

inline std::array<Foot, kNumFeet> allFeet() {
    return std::array<Foot, kNumFeet>{{FOOT_FL, FOOT_FR, FOOT_RL, FOOT_RR}};
}

struct FrameJacobian {
    MatrixX jacp;
    MatrixX jacr;
};

struct RobotState {
    VectorX qpos;
    VectorX qvel;
    Vector3 base_position;
    Eigen::Quaterniond base_quaternion;
};

struct ContactState {
    std::array<bool, kNumFeet> in_contact;

    ContactState() {
        in_contact.fill(false);
    }
};

}  // namespace go2wbc
