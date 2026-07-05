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
