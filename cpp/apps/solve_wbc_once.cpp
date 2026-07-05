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
