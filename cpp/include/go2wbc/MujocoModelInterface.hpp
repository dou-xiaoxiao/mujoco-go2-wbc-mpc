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
