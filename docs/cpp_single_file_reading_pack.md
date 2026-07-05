# C++ MPC/WBC Reading Pack

This is a single-file reading copy of the C++ implementation in `cpp/`.
It is meant for offline study: first read the guide below, then read the source
files in the listed order.

Generated from project-owned files only:

- `cpp/CMakeLists.txt`
- `cpp/include/go2wbc/*.hpp`
- `cpp/src/*.cpp`
- `cpp/apps/*.cpp`

Build outputs and third-party downloaded sources under `cpp/build*` are not
included.

## How To Read This File

The C++ stack follows the same control chain as the Python prototype:

```text
MuJoCo state
    -> model interface: M, h, B, foot positions, Jacobians
    -> centroidal / SRB MPC: desired contact forces
    -> full-body WBC QP: joint torques
    -> MuJoCo simulation step
```

You do not need to memorize every line. Read it in this order:

1. `Types.hpp`
   Basic aliases and foot indexing.
2. `MujocoModelInterface.hpp/.cpp`
   How MuJoCo exposes dynamics and kinematics.
3. `OsqpSolver.hpp/.cpp`
   How dense Eigen matrices are converted into OSQP data.
4. `CentroidalMpc.hpp/.cpp`
   The SRB-MPC contact-force planner.
5. `GeneralContactWbc.hpp/.cpp`
   The full-body WBC QP that computes joint torques.
6. `solve_mpc_once.cpp` and `solve_wbc_once.cpp`
   Small tests for one MPC/WBC solve.
7. `run_trot_rollout.cpp`
   The full closed-loop rollout: MPC -> WBC -> torque -> simulation.

## Control Interpretation

### Model Interface

`MujocoModelInterface` is the bridge between MuJoCo and the controller. It owns
`mjModel` and `mjData`, and provides:

- generalized dimensions: `nq`, `nv`, `nu`
- mass matrix `M(q)`
- bias term `h(q,v) = qfrc_bias - qfrc_passive`
- actuation matrix `B`
- foot positions
- foot Jacobians
- COM position, COM velocity, and approximate whole-body inertia

This is where the controller gets the physical quantities used in the QPs.

### Centroidal MPC

`CentroidalMpc` uses a simplified single-rigid-body model. It plans contact
forces for the four feet over a short horizon.

Conceptually:

```text
state:  COM position, COM velocity, body orientation, angular velocity
input:  foot contact forces
cost:   track desired COM/body motion and avoid excessive contact force
constraints: contact schedule, friction pyramid, nonnegative normal force
```

The output is a force reference for each foot. Swing feet are constrained to
zero force.

### Full-Body WBC

`GeneralContactWbc` solves the whole-body QP. It maps desired motion and MPC
force references into torque commands while respecting full floating-base
dynamics.

Conceptually:

```text
dynamics:      M(q) qddot + h(q,v) = B tau + Jc(q)^T f
stance task:   stance foot acceleration = 0
swing task:    swing foot acceleration follows the swing reference
force limits:  friction pyramid and positive normal force
torque limits: joint torque bounds
```

The output is the 12-dimensional joint torque command sent to MuJoCo.

### Rollout App

`run_trot_rollout.cpp` is the main closed-loop C++ demo. It:

1. reads the robot state from MuJoCo,
2. updates the trot contact phase,
3. solves MPC at a lower frequency,
4. solves WBC at a higher frequency,
5. applies torque to MuJoCo,
6. optionally records the trajectory as CSV.

The C++ route rollout is mainly a performance and architecture check. The
Python demo still has the more mature visual presentation tuning.

## Source Files

The following sections contain the complete project-owned C++ code.


---

## cpp\CMakeLists.txt

Build configuration: dependencies, include paths, libraries, and executable targets.

```cmake
cmake_minimum_required(VERSION 3.20)
project(go2_wbc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GO2WBC_USE_SYSTEM_EIGEN "Use a system-provided Eigen3 package" OFF)
option(GO2WBC_FETCH_EIGEN "Download Eigen if it is not installed" ON)
option(GO2WBC_FETCH_OSQP "Download OSQP if it is not installed" ON)

if(GO2WBC_USE_SYSTEM_EIGEN)
    find_package(Eigen3 QUIET)
endif()
if(NOT Eigen3_FOUND)
    if(GO2WBC_FETCH_EIGEN)
        include(FetchContent)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_GetProperties(eigen)
        if(NOT eigen_POPULATED)
            FetchContent_Populate(eigen)
        endif()
        if(NOT TARGET Eigen3::Eigen)
            add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
            target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
        endif()
    else()
        message(FATAL_ERROR "Eigen3 was not found. Install libeigen3-dev or enable GO2WBC_FETCH_EIGEN.")
    endif()
endif()

find_package(osqp QUIET)
if(NOT TARGET osqp::osqp AND NOT TARGET osqpstatic AND NOT TARGET osqp)
    if(GO2WBC_FETCH_OSQP)
        include(FetchContent)
        set(OSQP_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_DEMO_EXECUTABLE OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
        set(OSQP_BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
        set(OSQP_ENABLE_PRINTING OFF CACHE BOOL "" FORCE)
        set(OSQP_ENABLE_PROFILING OFF CACHE BOOL "" FORCE)
        FetchContent_Declare(
            osqp
            URL https://github.com/osqp/osqp/releases/download/v1.0.0/osqp-v1.0.0-src.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(osqp)
    else()
        message(FATAL_ERROR "OSQP was not found. Install OSQP or enable GO2WBC_FETCH_OSQP.")
    endif()
endif()

if(TARGET osqp::osqp)
    set(GO2WBC_OSQP_TARGET osqp::osqp)
elseif(TARGET osqpstatic)
    set(GO2WBC_OSQP_TARGET osqpstatic)
elseif(TARGET osqp)
    set(GO2WBC_OSQP_TARGET osqp)
else()
    message(FATAL_ERROR "Could not identify an OSQP CMake target.")
endif()

set(MUJOCO_ROOT "" CACHE PATH "Path to a MuJoCo SDK root containing include/ and lib/")
set(MUJOCO_PYTHON_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Lib/site-packages/mujoco" CACHE PATH
    "Optional path to a MuJoCo Python package directory")

find_path(MUJOCO_INCLUDE_DIR
    NAMES mujoco/mujoco.h
    PATHS
        "${MUJOCO_ROOT}/include"
        "${MUJOCO_ROOT}/include/mujoco"
        "${MUJOCO_PYTHON_ROOT}/include"
        "${MUJOCO_PYTHON_ROOT}/include/mujoco"
)

find_library(MUJOCO_LIBRARY
    NAMES mujoco libmujoco
    PATHS
        "${MUJOCO_ROOT}/lib"
        "${MUJOCO_ROOT}/bin"
        "${MUJOCO_PYTHON_ROOT}"
)

set(MUJOCO_DLL "${MUJOCO_PYTHON_ROOT}/mujoco.dll")
if(NOT MUJOCO_LIBRARY AND MINGW AND EXISTS "${MUJOCO_DLL}")
    find_program(GENDEF_EXECUTABLE gendef)
    find_program(DLLTOOL_EXECUTABLE dlltool)
    if(GENDEF_EXECUTABLE AND DLLTOOL_EXECUTABLE)
        set(MUJOCO_IMPORT_DIR "${CMAKE_BINARY_DIR}/mujoco_import")
        set(MUJOCO_DEF "${MUJOCO_IMPORT_DIR}/mujoco.def")
        set(MUJOCO_IMPORT_LIB "${MUJOCO_IMPORT_DIR}/libmujoco.dll.a")
        file(MAKE_DIRECTORY "${MUJOCO_IMPORT_DIR}")
        execute_process(
            COMMAND "${GENDEF_EXECUTABLE}" "${MUJOCO_DLL}"
            WORKING_DIRECTORY "${MUJOCO_IMPORT_DIR}"
            RESULT_VARIABLE GENDEF_RESULT
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(GENDEF_RESULT EQUAL 0 AND EXISTS "${MUJOCO_DEF}")
            execute_process(
                COMMAND "${DLLTOOL_EXECUTABLE}" -d "${MUJOCO_DEF}" -l "${MUJOCO_IMPORT_LIB}" -D mujoco.dll
                RESULT_VARIABLE DLLTOOL_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(DLLTOOL_RESULT EQUAL 0 AND EXISTS "${MUJOCO_IMPORT_LIB}")
                set(MUJOCO_LIBRARY "${MUJOCO_IMPORT_LIB}" CACHE FILEPATH "MuJoCo MinGW import library" FORCE)
            endif()
        endif()
    endif()
endif()

if(NOT MUJOCO_INCLUDE_DIR)
    message(FATAL_ERROR "MuJoCo headers were not found. Set MUJOCO_ROOT or MUJOCO_PYTHON_ROOT.")
endif()

if(NOT MUJOCO_LIBRARY)
    message(FATAL_ERROR "MuJoCo library was not found. Set MUJOCO_ROOT to an installed MuJoCo SDK.")
endif()

add_library(go2wbc_core
    src/CentroidalMpc.cpp
    src/GeneralContactWbc.cpp
    src/MujocoModelInterface.cpp
    src/OsqpSolver.cpp
)

target_include_directories(go2wbc_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(go2wbc_core
    PUBLIC
        Eigen3::Eigen
        ${MUJOCO_LIBRARY}
        ${GO2WBC_OSQP_TARGET}
)

add_executable(inspect_dynamics
    apps/inspect_dynamics.cpp
)

target_link_libraries(inspect_dynamics PRIVATE go2wbc_core)

add_executable(solve_wbc_once
    apps/solve_wbc_once.cpp
)

target_link_libraries(solve_wbc_once PRIVATE go2wbc_core)

add_executable(solve_mpc_once
    apps/solve_mpc_once.cpp
)

target_link_libraries(solve_mpc_once PRIVATE go2wbc_core)

add_executable(run_trot_rollout
    apps/run_trot_rollout.cpp
)

target_link_libraries(run_trot_rollout PRIVATE go2wbc_core)

if(WIN32 AND EXISTS "${MUJOCO_DLL}")
    add_custom_command(TARGET inspect_dynamics POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:inspect_dynamics>/mujoco.dll"
    )
    add_custom_command(TARGET solve_wbc_once POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:solve_wbc_once>/mujoco.dll"
    )
    add_custom_command(TARGET solve_mpc_once POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:solve_mpc_once>/mujoco.dll"
    )
    add_custom_command(TARGET run_trot_rollout POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MUJOCO_DLL}"
            "$<TARGET_FILE_DIR:run_trot_rollout>/mujoco.dll"
    )
endif()
```


---

## cpp\include\go2wbc\Types.hpp

Basic Eigen aliases, foot enum, foot order, and helper functions.

```cpp
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
```


---

## cpp\include\go2wbc\WbcTypes.hpp

Small shared WBC result/config types.

```cpp
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
```


---

## cpp\include\go2wbc\MujocoModelInterface.hpp

Public interface for MuJoCo dynamics and kinematics access.

```cpp
#pragma once

#include <array>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "go2wbc/Types.hpp"

namespace go2wbc {

class MujocoModelInterface {
public:
    explicit MujocoModelInterface(const std::string& model_path);
    ~MujocoModelInterface();

    MujocoModelInterface(const MujocoModelInterface&) = delete;
    MujocoModelInterface& operator=(const MujocoModelInterface&) = delete;

    int nq() const;
    int nv() const;
    int nu() const;

    void setKeyframe(const std::string& name);
    void forward();

    VectorX qpos() const;
    VectorX qvel() const;
    RobotState state() const;

    MatrixX massMatrix() const;
    VectorX passiveForces() const;
    VectorX biasForces(bool include_passive) const;

    MatrixX actuationMatrix() const;
    MatrixX actuationMatrixUncached();
    double checkActuationMatrixCache();

    double totalMass() const;
    Vector3 centerOfMass() const;
    MatrixX compositeInertiaWorldAboutCom() const;
    Vector3 basePosition() const;
    Vector3 baseLinearVelocity() const;
    Vector3 baseAngularVelocity() const;
    MatrixX baseRotationWorldFromBase() const;
    MatrixX baseRotationBaseFromWorld() const;

    Vector3 worldVectorToBase(const Vector3& vector_world) const;
    Vector3 baseVectorToWorld(const Vector3& vector_base) const;
    Vector3 worldPointToBase(const Vector3& point_world) const;
    Vector3 basePointToWorld(const Vector3& point_base) const;

    FrameJacobian geomJacobian(const std::string& geom_name) const;
    Vector3 geomCenterPosition(const std::string& geom_name) const;
    Vector3 geomPosition(const std::string& geom_name) const;
    Vector3 geomVelocity(const std::string& geom_name) const;
    bool geomHasContact(const std::string& geom_name) const;
    double geomContactRadius(const std::string& geom_name) const;

    MatrixX stackedGeomJacobian(const std::vector<std::string>& geom_names) const;

    mjModel* model();
    mjData* data();
    const mjModel* model() const;
    const mjData* data() const;

private:
    int geomId(const std::string& name) const;
    int bodyId(const std::string& name) const;
    int keyId(const std::string& name) const;
    bool usesFootContactPoint(const std::string& geom_name, int geom_id) const;

    std::string model_path_;
    mjModel* model_;
    mjData* data_;
    int base_body_id_;
    MatrixX actuation_matrix_cached_;
};

}  // namespace go2wbc
```


---

## cpp\src\MujocoModelInterface.cpp

Implementation of mass matrix, bias forces, actuation matrix, Jacobians, COM, and simulation stepping.

```cpp
#include "go2wbc/MujocoModelInterface.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace go2wbc {

namespace {

static const char* kBaseBodyName = "base";

bool isFootName(const std::string& name) {
    return name == "FL" || name == "FR" || name == "RL" || name == "RR";
}

VectorX copyVector(const double* data, int size) {
    VectorX out(size);
    for (int i = 0; i < size; ++i) {
        out(i) = data[i];
    }
    return out;
}

}  // namespace

MujocoModelInterface::MujocoModelInterface(const std::string& model_path)
    : model_path_(model_path),
      model_(0),
      data_(0),
      base_body_id_(-1) {
    char error[1024];
    std::memset(error, 0, sizeof(error));

    model_ = mj_loadXML(model_path.c_str(), 0, error, sizeof(error));
    if (model_ == 0) {
        throw std::runtime_error(std::string("Failed to load MuJoCo model: ") + error);
    }

    data_ = mj_makeData(model_);
    if (data_ == 0) {
        mj_deleteModel(model_);
        model_ = 0;
        throw std::runtime_error("Failed to allocate MuJoCo data.");
    }

    base_body_id_ = bodyId(kBaseBodyName);
    mj_forward(model_, data_);
    actuation_matrix_cached_ = actuationMatrixUncached();
}

MujocoModelInterface::~MujocoModelInterface() {
    if (data_ != 0) {
        mj_deleteData(data_);
        data_ = 0;
    }
    if (model_ != 0) {
        mj_deleteModel(model_);
        model_ = 0;
    }
}

int MujocoModelInterface::nq() const { return model_->nq; }
int MujocoModelInterface::nv() const { return model_->nv; }
int MujocoModelInterface::nu() const { return model_->nu; }

void MujocoModelInterface::setKeyframe(const std::string& name) {
    int id = keyId(name);
    const double* key_qpos = model_->key_qpos + id * model_->nq;
    for (int i = 0; i < model_->nq; ++i) {
        data_->qpos[i] = key_qpos[i];
    }
    for (int i = 0; i < model_->nv; ++i) {
        data_->qvel[i] = 0.0;
        data_->qacc[i] = 0.0;
    }
    if (model_->nu > 0) {
        const double* key_ctrl = model_->key_ctrl + id * model_->nu;
        for (int i = 0; i < model_->nu; ++i) {
            data_->ctrl[i] = key_ctrl[i];
        }
    }
    mj_forward(model_, data_);
}

void MujocoModelInterface::forward() {
    mj_forward(model_, data_);
}

VectorX MujocoModelInterface::qpos() const {
    return copyVector(data_->qpos, model_->nq);
}

VectorX MujocoModelInterface::qvel() const {
    return copyVector(data_->qvel, model_->nv);
}

RobotState MujocoModelInterface::state() const {
    RobotState out;
    out.qpos = qpos();
    out.qvel = qvel();
    out.base_position = basePosition();
    out.base_quaternion = Eigen::Quaterniond(
        data_->qpos[3],
        data_->qpos[4],
        data_->qpos[5],
        data_->qpos[6]
    );
    return out;
}

MatrixX MujocoModelInterface::massMatrix() const {
    MatrixX mass(model_->nv, model_->nv);
    mass.setZero();
    mj_fullM(model_, data_, mass.data());
    return mass;
}

VectorX MujocoModelInterface::passiveForces() const {
    return copyVector(data_->qfrc_passive, model_->nv);
}

VectorX MujocoModelInterface::biasForces(bool include_passive) const {
    VectorX bias = copyVector(data_->qfrc_bias, model_->nv);
    if (include_passive) {
        return bias;
    }
    VectorX passive = copyVector(data_->qfrc_passive, model_->nv);
    return bias - passive;
}

MatrixX MujocoModelInterface::actuationMatrix() const {
    return actuation_matrix_cached_;
}

MatrixX MujocoModelInterface::actuationMatrixUncached() {
    VectorX original_ctrl = copyVector(data_->ctrl, model_->nu);
    VectorX original_qacc = copyVector(data_->qacc, model_->nv);

    MatrixX matrix(model_->nv, model_->nu);
    matrix.setZero();

    for (int i = 0; i < model_->nu; ++i) {
        data_->ctrl[i] = 0.0;
    }
    mj_forward(model_, data_);

    for (int actuator = 0; actuator < model_->nu; ++actuator) {
        for (int i = 0; i < model_->nu; ++i) {
            data_->ctrl[i] = 0.0;
        }
        data_->ctrl[actuator] = 1.0;
        mj_forward(model_, data_);
        for (int row = 0; row < model_->nv; ++row) {
            matrix(row, actuator) = data_->qfrc_actuator[row];
        }
    }

    for (int i = 0; i < model_->nu; ++i) {
        data_->ctrl[i] = original_ctrl(i);
    }
    for (int i = 0; i < model_->nv; ++i) {
        data_->qacc[i] = original_qacc(i);
    }
    mj_forward(model_, data_);
    return matrix;
}

double MujocoModelInterface::checkActuationMatrixCache() {
    MatrixX uncached = actuationMatrixUncached();
    MatrixX diff = actuation_matrix_cached_ - uncached;
    return diff.cwiseAbs().maxCoeff();
}

double MujocoModelInterface::totalMass() const {
    double mass = 0.0;
    for (int body = 0; body < model_->nbody; ++body) {
        mass += model_->body_mass[body];
    }
    return mass;
}

Vector3 MujocoModelInterface::centerOfMass() const {
    const double* com = data_->subtree_com + 3 * base_body_id_;
    return Vector3(com[0], com[1], com[2]);
}

MatrixX MujocoModelInterface::compositeInertiaWorldAboutCom() const {
    Vector3 com = centerOfMass();
    MatrixX inertia = MatrixX::Zero(3, 3);

    for (int body = 1; body < model_->nbody; ++body) {
        double mass = model_->body_mass[body];
        if (mass <= 0.0) {
            continue;
        }

        MatrixX body_inertia = MatrixX::Zero(3, 3);
        body_inertia(0, 0) = model_->body_inertia[3 * body + 0];
        body_inertia(1, 1) = model_->body_inertia[3 * body + 1];
        body_inertia(2, 2) = model_->body_inertia[3 * body + 2];

        MatrixX rotation_world_from_body(3, 3);
        const double* xmat = data_->xmat + 9 * body;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                rotation_world_from_body(row, col) = xmat[3 * row + col];
            }
        }

        double mat_raw[9];
        mju_quat2Mat(mat_raw, model_->body_iquat + 4 * body);
        MatrixX rotation_body_from_inertia(3, 3);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                rotation_body_from_inertia(row, col) = mat_raw[3 * row + col];
            }
        }

        MatrixX rotation_world_from_inertia = rotation_world_from_body * rotation_body_from_inertia;
        MatrixX body_inertia_world =
            rotation_world_from_inertia * body_inertia * rotation_world_from_inertia.transpose();

        const double* xipos = data_->xipos + 3 * body;
        Vector3 r(xipos[0] - com(0), xipos[1] - com(1), xipos[2] - com(2));
        inertia += body_inertia_world
            + mass * (r.dot(r) * MatrixX::Identity(3, 3) - r * r.transpose());
    }

    return inertia;
}

Vector3 MujocoModelInterface::basePosition() const {
    const double* p = data_->xpos + 3 * base_body_id_;
    return Vector3(p[0], p[1], p[2]);
}

Vector3 MujocoModelInterface::baseLinearVelocity() const {
    return Vector3(data_->qvel[0], data_->qvel[1], data_->qvel[2]);
}

Vector3 MujocoModelInterface::baseAngularVelocity() const {
    return Vector3(data_->qvel[3], data_->qvel[4], data_->qvel[5]);
}

MatrixX MujocoModelInterface::baseRotationWorldFromBase() const {
    MatrixX rotation(3, 3);
    const double* xmat = data_->xmat + 9 * base_body_id_;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            rotation(row, col) = xmat[3 * row + col];
        }
    }
    return rotation;
}

MatrixX MujocoModelInterface::baseRotationBaseFromWorld() const {
    return baseRotationWorldFromBase().transpose();
}

Vector3 MujocoModelInterface::worldVectorToBase(const Vector3& vector_world) const {
    return baseRotationBaseFromWorld() * vector_world;
}

Vector3 MujocoModelInterface::baseVectorToWorld(const Vector3& vector_base) const {
    return baseRotationWorldFromBase() * vector_base;
}

Vector3 MujocoModelInterface::worldPointToBase(const Vector3& point_world) const {
    return worldVectorToBase(point_world - basePosition());
}

Vector3 MujocoModelInterface::basePointToWorld(const Vector3& point_base) const {
    return basePosition() + baseVectorToWorld(point_base);
}

FrameJacobian MujocoModelInterface::geomJacobian(const std::string& geom_name) const {
    int id = geomId(geom_name);
    FrameJacobian jac;
    jac.jacp = MatrixX::Zero(3, model_->nv);
    jac.jacr = MatrixX::Zero(3, model_->nv);
    mj_jacGeom(model_, data_, jac.jacp.data(), jac.jacr.data(), id);
    return jac;
}

Vector3 MujocoModelInterface::geomCenterPosition(const std::string& geom_name) const {
    int id = geomId(geom_name);
    const double* p = data_->geom_xpos + 3 * id;
    return Vector3(p[0], p[1], p[2]);
}

Vector3 MujocoModelInterface::geomPosition(const std::string& geom_name) const {
    int id = geomId(geom_name);
    Vector3 p = geomCenterPosition(geom_name);
    if (usesFootContactPoint(geom_name, id)) {
        p(2) -= model_->geom_size[3 * id + 0];
    }
    return p;
}

Vector3 MujocoModelInterface::geomVelocity(const std::string& geom_name) const {
    FrameJacobian jac = geomJacobian(geom_name);
    return jac.jacp * qvel();
}

bool MujocoModelInterface::geomHasContact(const std::string& geom_name) const {
    int id = geomId(geom_name);
    for (int contact = 0; contact < data_->ncon; ++contact) {
        const mjContact& c = data_->contact[contact];
        if (c.geom1 == id || c.geom2 == id) {
            return true;
        }
    }
    return false;
}

double MujocoModelInterface::geomContactRadius(const std::string& geom_name) const {
    int id = geomId(geom_name);
    if (!usesFootContactPoint(geom_name, id)) {
        return 0.0;
    }
    return model_->geom_size[3 * id + 0];
}

MatrixX MujocoModelInterface::stackedGeomJacobian(const std::vector<std::string>& geom_names) const {
    MatrixX stacked(3 * static_cast<int>(geom_names.size()), model_->nv);
    for (int i = 0; i < static_cast<int>(geom_names.size()); ++i) {
        FrameJacobian jac = geomJacobian(geom_names[i]);
        stacked.block(3 * i, 0, 3, model_->nv) = jac.jacp;
    }
    return stacked;
}

mjModel* MujocoModelInterface::model() { return model_; }
mjData* MujocoModelInterface::data() { return data_; }
const mjModel* MujocoModelInterface::model() const { return model_; }
const mjData* MujocoModelInterface::data() const { return data_; }

int MujocoModelInterface::geomId(const std::string& name) const {
    int id = mj_name2id(model_, mjOBJ_GEOM, name.c_str());
    if (id < 0) {
        throw std::runtime_error("Unknown geom: " + name);
    }
    return id;
}

int MujocoModelInterface::bodyId(const std::string& name) const {
    int id = mj_name2id(model_, mjOBJ_BODY, name.c_str());
    if (id < 0) {
        throw std::runtime_error("Unknown body: " + name);
    }
    return id;
}

int MujocoModelInterface::keyId(const std::string& name) const {
    int id = mj_name2id(model_, mjOBJ_KEY, name.c_str());
    if (id < 0) {
        throw std::runtime_error("Unknown keyframe: " + name);
    }
    return id;
}

bool MujocoModelInterface::usesFootContactPoint(const std::string& geom_name, int geom_id) const {
    return isFootName(geom_name)
        && model_->geom_type[geom_id] == mjGEOM_SPHERE
        && model_->geom_size[3 * geom_id + 0] > 0.0;
}

}  // namespace go2wbc
```


---

## cpp\include\go2wbc\OsqpSolver.hpp

Small wrapper interface around the OSQP C API.

```cpp
#pragma once

#include <string>
#include <vector>

#include <osqp.h>

#include "go2wbc/Types.hpp"

namespace go2wbc {

struct SparseCSC {
    int rows;
    int cols;
    std::vector<OSQPInt> col_ptr;
    std::vector<OSQPInt> row_idx;
    std::vector<OSQPFloat> values;

    SparseCSC() : rows(0), cols(0) {}
};

struct QpProblem {
    SparseCSC P;
    VectorX q;
    SparseCSC A;
    VectorX lower;
    VectorX upper;
};

struct QpSolution {
    VectorX x;
    std::string status;
    int status_value;
    double objective;
    int iterations;

    QpSolution() : status_value(0), objective(0.0), iterations(0) {}
};

class OsqpSolver {
public:
    OsqpSolver();
    ~OsqpSolver();

    OsqpSolver(const OsqpSolver&) = delete;
    OsqpSolver& operator=(const OsqpSolver&) = delete;

    void setTolerances(double eps_abs, double eps_rel);
    void setMaxIterations(int max_iter);
    void setPolishing(bool enabled);
    QpSolution solve(const QpProblem& problem);

private:
    bool sameStructure(const QpProblem& problem) const;
    void setup(const QpProblem& problem);
    void update(const QpProblem& problem);
    void cleanup();
    void storeProblemData(const QpProblem& problem);

    OSQPSolver* solver_;
    OSQPCscMatrix p_csc_;
    OSQPCscMatrix a_csc_;

    SparseCSC p_data_;
    SparseCSC a_data_;
    std::vector<OSQPFloat> q_data_;
    std::vector<OSQPFloat> l_data_;
    std::vector<OSQPFloat> u_data_;
    VectorX last_x_;

    int n_;
    int m_;
    int p_nnz_;
    int a_nnz_;
    double eps_abs_;
    double eps_rel_;
    int max_iter_;
    bool polishing_;
};

SparseCSC denseToCSC(const MatrixX& dense, bool upper_triangle_only);

}  // namespace go2wbc
```


---

## cpp\src\OsqpSolver.cpp

Conversion from Eigen dense matrices to OSQP CSC data and solve result unpacking.

```cpp
#include "go2wbc/OsqpSolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace go2wbc {

namespace {

const double kSparseTolerance = 1.0e-12;

std::vector<OSQPFloat> copyEigenVector(const VectorX& vector) {
    std::vector<OSQPFloat> out(static_cast<size_t>(vector.size()));
    for (int i = 0; i < vector.size(); ++i) {
        double value = vector(i);
        if (std::isinf(value) && value < 0.0) {
            value = -OSQP_INFTY;
        } else if (std::isinf(value) && value > 0.0) {
            value = OSQP_INFTY;
        }
        out[static_cast<size_t>(i)] = static_cast<OSQPFloat>(value);
    }
    return out;
}

}  // namespace

SparseCSC denseToCSC(const MatrixX& dense, bool upper_triangle_only) {
    SparseCSC out;
    out.rows = static_cast<int>(dense.rows());
    out.cols = static_cast<int>(dense.cols());
    out.col_ptr.resize(static_cast<size_t>(out.cols + 1), 0);

    for (int col = 0; col < out.cols; ++col) {
        out.col_ptr[static_cast<size_t>(col)] = static_cast<OSQPInt>(out.values.size());
        int row_start = upper_triangle_only ? 0 : 0;
        int row_end = upper_triangle_only ? std::min(col, out.rows - 1) : out.rows - 1;
        for (int row = row_start; row <= row_end; ++row) {
            double value = dense(row, col);
            if (std::abs(value) > kSparseTolerance) {
                out.row_idx.push_back(static_cast<OSQPInt>(row));
                out.values.push_back(static_cast<OSQPFloat>(value));
            }
        }
    }
    out.col_ptr[static_cast<size_t>(out.cols)] = static_cast<OSQPInt>(out.values.size());
    return out;
}

OsqpSolver::OsqpSolver()
    : solver_(0),
      n_(0),
      m_(0),
      p_nnz_(0),
      a_nnz_(0),
      eps_abs_(1.0e-6),
      eps_rel_(1.0e-6),
      max_iter_(10000),
      polishing_(true) {}

OsqpSolver::~OsqpSolver() {
    cleanup();
}

void OsqpSolver::setTolerances(double eps_abs, double eps_rel) {
    eps_abs_ = eps_abs;
    eps_rel_ = eps_rel;
}

void OsqpSolver::setMaxIterations(int max_iter) {
    max_iter_ = max_iter;
}

void OsqpSolver::setPolishing(bool enabled) {
    polishing_ = enabled;
}

QpSolution OsqpSolver::solve(const QpProblem& problem) {
    if (problem.P.rows != problem.P.cols) {
        throw std::runtime_error("OSQP requires a square P matrix.");
    }
    if (problem.A.cols != problem.P.cols) {
        throw std::runtime_error("A and P have inconsistent variable dimensions.");
    }
    if (problem.q.size() != problem.P.cols ||
        problem.lower.size() != problem.A.rows ||
        problem.upper.size() != problem.A.rows) {
        throw std::runtime_error("QP vector dimensions are inconsistent.");
    }

    if (solver_ == 0 || !sameStructure(problem)) {
        setup(problem);
    } else {
        update(problem);
    }

    if (last_x_.size() == problem.q.size()) {
        osqp_warm_start(solver_, last_x_.data(), OSQP_NULL);
    }

    OSQPInt flag = osqp_solve(solver_);
    if (flag != 0) {
        throw std::runtime_error("osqp_solve failed with flag " + std::to_string(static_cast<int>(flag)));
    }

    QpSolution solution;
    solution.x = VectorX::Zero(problem.q.size());
    if (solver_->solution != 0 && solver_->solution->x != 0) {
        for (int i = 0; i < solution.x.size(); ++i) {
            solution.x(i) = solver_->solution->x[i];
        }
        last_x_ = solution.x;
    }
    if (solver_->info != 0) {
        solution.status = solver_->info->status;
        solution.status_value = static_cast<int>(solver_->info->status_val);
        solution.objective = solver_->info->obj_val;
        solution.iterations = static_cast<int>(solver_->info->iter);
    }
    return solution;
}

bool OsqpSolver::sameStructure(const QpProblem& problem) const {
    return problem.P.cols == n_
        && problem.A.rows == m_
        && static_cast<int>(problem.P.values.size()) == p_nnz_
        && static_cast<int>(problem.A.values.size()) == a_nnz_
        && problem.P.col_ptr == p_data_.col_ptr
        && problem.P.row_idx == p_data_.row_idx
        && problem.A.col_ptr == a_data_.col_ptr
        && problem.A.row_idx == a_data_.row_idx;
}

void OsqpSolver::setup(const QpProblem& problem) {
    cleanup();
    storeProblemData(problem);

    OSQPSettings settings;
    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.warm_starting = 1;
    settings.polishing = polishing_ ? 1 : 0;
    settings.eps_abs = eps_abs_;
    settings.eps_rel = eps_rel_;
    settings.max_iter = max_iter_;

    OSQPCscMatrix_set_data(
        &p_csc_,
        p_data_.rows,
        p_data_.cols,
        static_cast<OSQPInt>(p_data_.values.size()),
        p_data_.values.data(),
        p_data_.row_idx.data(),
        p_data_.col_ptr.data()
    );
    OSQPCscMatrix_set_data(
        &a_csc_,
        a_data_.rows,
        a_data_.cols,
        static_cast<OSQPInt>(a_data_.values.size()),
        a_data_.values.data(),
        a_data_.row_idx.data(),
        a_data_.col_ptr.data()
    );

    OSQPInt flag = osqp_setup(
        &solver_,
        &p_csc_,
        q_data_.data(),
        &a_csc_,
        l_data_.data(),
        u_data_.data(),
        static_cast<OSQPInt>(m_),
        static_cast<OSQPInt>(n_),
        &settings
    );
    if (flag != 0) {
        cleanup();
        throw std::runtime_error("osqp_setup failed with flag " + std::to_string(static_cast<int>(flag)));
    }
}

void OsqpSolver::update(const QpProblem& problem) {
    storeProblemData(problem);
    OSQPInt vec_flag = osqp_update_data_vec(solver_, q_data_.data(), l_data_.data(), u_data_.data());
    if (vec_flag != 0) {
        setup(problem);
        return;
    }
    OSQPInt mat_flag = osqp_update_data_mat(
        solver_,
        p_data_.values.data(),
        OSQP_NULL,
        static_cast<OSQPInt>(p_data_.values.size()),
        a_data_.values.data(),
        OSQP_NULL,
        static_cast<OSQPInt>(a_data_.values.size())
    );
    if (mat_flag != 0) {
        setup(problem);
    }
}

void OsqpSolver::cleanup() {
    if (solver_ != 0) {
        osqp_cleanup(solver_);
        solver_ = 0;
    }
}

void OsqpSolver::storeProblemData(const QpProblem& problem) {
    p_data_ = problem.P;
    a_data_ = problem.A;
    q_data_ = copyEigenVector(problem.q);
    l_data_ = copyEigenVector(problem.lower);
    u_data_ = copyEigenVector(problem.upper);
    n_ = problem.P.cols;
    m_ = problem.A.rows;
    p_nnz_ = static_cast<int>(problem.P.values.size());
    a_nnz_ = static_cast<int>(problem.A.values.size());
}

}  // namespace go2wbc
```


---

## cpp\include\go2wbc\CentroidalMpc.hpp

MPC config, input, output, and class interface.

```cpp
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
```


---

## cpp\src\CentroidalMpc.cpp

SRB-MPC QP assembly: state rollout, contact-force variables, costs, dynamics, and friction/contact constraints.

```cpp
#include "go2wbc/CentroidalMpc.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

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
    solver_.setTolerances(1.0e-7, 1.0e-7);
    solver_.setMaxIterations(10000);
    solver_.setPolishing(true);
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

    MatrixX P = MatrixX::Zero(nvar, nvar);
    VectorX q = VectorX::Zero(nvar);

    for (int step = 1; step <= n_steps; ++step) {
        int base = step * kStateDim;
        for (int axis = 0; axis < 3; ++axis) {
            P(base + axis, base + axis) += config_.weight_com_position;
            q(base + axis) += -config_.weight_com_position * com_pos_ref(step, axis);
            P(base + 3 + axis, base + 3 + axis) += config_.weight_com_velocity;
            q(base + 3 + axis) += -config_.weight_com_velocity * com_vel_ref(step, axis);
            P(base + 6 + axis, base + 6 + axis) += config_.weight_orientation;
            q(base + 6 + axis) += -config_.weight_orientation * ori_ref(step, axis);
            P(base + 9 + axis, base + 9 + axis) += config_.weight_angular_velocity;
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
                P(base + axis, base + axis) += config_.weight_force_regularization;
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
                P(prev + idx, prev + idx) += config_.weight_force_rate;
                P(curr + idx, curr + idx) += config_.weight_force_rate;
                P(prev + idx, curr + idx) += -config_.weight_force_rate;
                P(curr + idx, prev + idx) += -config_.weight_force_rate;
            }
        }
    }

    for (int i = 0; i < nvar; ++i) {
        P(i, i) += 1.0e-9;
    }

    int dyn_rows = (n_steps + 1) * kStateDim;
    int force_rows = n_steps * kNumFeet * 5;
    int ncon = dyn_rows + force_rows;
    MatrixX A = MatrixX::Zero(ncon, nvar);
    VectorX lower = VectorX::Zero(ncon);
    VectorX upper = VectorX::Zero(ncon);
    int row = 0;

    for (int idx = 0; idx < kStateDim; ++idx) {
        A(row, idx) = 1.0;
        lower(row) = x0(idx);
        upper(row) = x0(idx);
        row++;
    }

    for (int step = 0; step < n_steps; ++step) {
        int xk = step * kStateDim;
        int xkp1 = (step + 1) * kStateDim;
        int uk = n_state_vars + step * kForceDimAll;

        for (int axis = 0; axis < 3; ++axis) {
            A(row, xkp1 + axis) = 1.0;
            A(row, xk + axis) = -1.0;
            A(row, xk + 3 + axis) = -config_.dt;
            lower(row) = 0.0;
            upper(row) = 0.0;
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            A(row, xkp1 + 3 + axis) = 1.0;
            A(row, xk + 3 + axis) = -1.0;
            for (int force_axis = axis; force_axis < kForceDimAll; force_axis += 3) {
                A(row, uk + force_axis) = -config_.dt / mass;
            }
            lower(row) = config_.dt * gravity(axis);
            upper(row) = config_.dt * gravity(axis);
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            A(row, xkp1 + 6 + axis) = 1.0;
            A(row, xk + 6 + axis) = -1.0;
            A(row, xk + 9 + axis) = -config_.dt;
            lower(row) = 0.0;
            upper(row) = 0.0;
            row++;
        }

        for (int axis = 0; axis < 3; ++axis) {
            A(row, xkp1 + 9 + axis) = 1.0;
            A(row, xk + 9 + axis) = -1.0;
            for (int force_idx = 0; force_idx < kForceDimAll; ++force_idx) {
                A(row, uk + force_idx) = -config_.dt * angular_map(axis, force_idx);
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

            A(row, fx) = 1.0;
            A(row, fz) = -config_.friction_mu;
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            A(row, fx) = -1.0;
            A(row, fz) = -config_.friction_mu;
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            A(row, fy) = 1.0;
            A(row, fz) = -config_.friction_mu;
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            A(row, fy) = -1.0;
            A(row, fz) = -config_.friction_mu;
            lower(row) = stance ? -kInf : 0.0;
            upper(row) = 0.0;
            row++;

            A(row, fz) = 1.0;
            lower(row) = stance ? config_.normal_force_min : 0.0;
            upper(row) = stance ? kInf : 0.0;
            row++;
        }
    }

    QpProblem problem;
    problem.P = denseToCSC(P, true);
    problem.q = q;
    problem.A = denseToCSC(A, false);
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
```


---

## cpp\include\go2wbc\GeneralContactWbc.hpp

WBC config, input, output, and class interface.

```cpp
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
```


---

## cpp\src\GeneralContactWbc.cpp

Full-body WBC QP assembly: dynamics constraint, stance constraints, swing task cost, force cost, friction constraints, torque limits.

```cpp
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
```


---

## cpp\apps\inspect_dynamics.cpp

Small executable for checking MuJoCo dimensions and dynamics quantities.

```cpp
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "go2wbc/MujocoModelInterface.hpp"

namespace {

template <typename Function>
double benchmarkMilliseconds(const std::string& name, int repeats, Function function) {
    volatile double sink = 0.0;
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        sink += function();
    }
    const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double mean_ms = elapsed_ms / static_cast<double>(repeats);
    std::cout << std::setw(24) << name << ": " << std::fixed << std::setprecision(4)
              << mean_ms << " ms/call" << std::endl;
    return sink;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: inspect_dynamics <path/to/scene.xml>" << std::endl;
            return 2;
        }

        const std::string model_path = argv[1];
        go2wbc::MujocoModelInterface robot(model_path);
        robot.setKeyframe("home");

        std::cout << "Loaded model: " << model_path << std::endl;
        std::cout << "nq=" << robot.nq() << " nv=" << robot.nv() << " nu=" << robot.nu() << std::endl;
        std::cout << "total_mass=" << robot.totalMass() << std::endl;
        std::cout << "B_cache_error=" << robot.checkActuationMatrixCache() << std::endl;
        std::cout << "base_position=" << robot.basePosition().transpose() << std::endl;

        const std::array<go2wbc::Foot, go2wbc::kNumFeet> feet = go2wbc::allFeet();
        for (int i = 0; i < go2wbc::kNumFeet; ++i) {
            const char* name = go2wbc::footName(feet[i]);
            std::cout << "foot " << name << " position=" << robot.geomPosition(name).transpose() << std::endl;
        }

        std::vector<std::string> foot_names;
        foot_names.push_back("FL");
        foot_names.push_back("FR");
        foot_names.push_back("RL");
        foot_names.push_back("RR");

        const int repeats = 1000;
        std::cout << "\nBenchmark, repeats=" << repeats << std::endl;

        benchmarkMilliseconds("massMatrix", repeats, [&robot]() {
            go2wbc::MatrixX m = robot.massMatrix();
            return m(0, 0);
        });

        benchmarkMilliseconds("biasForces", repeats, [&robot]() {
            go2wbc::VectorX h = robot.biasForces(false);
            return h(0);
        });

        benchmarkMilliseconds("actuationMatrix cached", repeats, [&robot]() {
            go2wbc::MatrixX b = robot.actuationMatrix();
            return b(6, 0);
        });

        benchmarkMilliseconds("stackedGeomJacobian", repeats, [&robot, &foot_names]() {
            go2wbc::MatrixX j = robot.stackedGeomJacobian(foot_names);
            return j(0, 0);
        });

        benchmarkMilliseconds("foot positions", repeats, [&robot]() {
            double value = 0.0;
            value += robot.geomPosition("FL")(0);
            value += robot.geomPosition("FR")(0);
            value += robot.geomPosition("RL")(0);
            value += robot.geomPosition("RR")(0);
            return value;
        });

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
```


---

## cpp\apps\solve_mpc_once.cpp

Small executable for solving one MPC problem and inspecting force output.

```cpp
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
```


---

## cpp\apps\solve_wbc_once.cpp

Small executable for solving one WBC problem and inspecting torque/residual output.

```cpp
#include <algorithm>
#include <iostream>
#include <string>

#include "go2wbc/GeneralContactWbc.hpp"

using go2wbc::FOOT_FL;
using go2wbc::FOOT_FR;
using go2wbc::FOOT_RL;
using go2wbc::FOOT_RR;
using go2wbc::GeneralContactWbc;
using go2wbc::GeneralContactWbcConfig;
using go2wbc::GeneralContactWbcInput;
using go2wbc::MujocoModelInterface;
using go2wbc::VectorX;

double maxAbs(const VectorX& v) {
    if (v.size() == 0) {
        return 0.0;
    }
    return v.cwiseAbs().maxCoeff();
}

int main(int argc, char** argv) {
    try {
        std::string model_path = "../models/mujoco_menagerie/unitree_go2/scene.xml";
        if (argc >= 2) {
            model_path = argv[1];
        }

        MujocoModelInterface robot(model_path);
        robot.setKeyframe("home");

        GeneralContactWbcConfig config;
        config.stance_feet = {FOOT_FL, FOOT_FR, FOOT_RL, FOOT_RR};
        config.swing_feet = {};
        config.weight_force = 1.0;
        config.kp_stance = 100.0;
        config.kd_stance = 20.0;
        config.use_jdot_v = false;

        GeneralContactWbcInput input;
        input.qpos_ref = robot.qpos();
        input.force_ref = VectorX();
        input.force_zero_weights = VectorX();
        for (go2wbc::Foot foot : go2wbc::allFeet()) {
            input.stance_refs[static_cast<size_t>(foot)].enabled = true;
            input.stance_refs[static_cast<size_t>(foot)].position = robot.geomPosition(go2wbc::footName(foot));
        }

        GeneralContactWbc wbc(config);
        go2wbc::GeneralContactWbcOutput out = wbc.solve(robot, input);

        std::cout << "status=" << out.status
                  << " iter=" << out.iterations
                  << " obj=" << out.objective << "\n";
        std::cout << "tau_max=" << maxAbs(out.tau)
                  << " dyn_res=" << maxAbs(out.dynamics_residual)
                  << " stance_res=" << maxAbs(out.stance_residual) << "\n";
        std::cout << "tau=" << out.tau.transpose() << "\n";
        std::cout << "forces=" << out.contact_forces.transpose() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```


---

## cpp\apps\run_trot_rollout.cpp

Closed-loop rollout: contact schedule, swing references, MPC, WBC, torque application, timing, and CSV recording.

```cpp
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
```

