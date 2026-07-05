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
