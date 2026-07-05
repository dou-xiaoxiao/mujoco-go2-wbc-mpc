# C++ MPC/WBC Implementation

This directory contains a plain C++ implementation of the MuJoCo Go2
locomotion control stack. The Python stack remains the recommended visual demo,
while the C++ stack is used to test the same control ideas with lower runtime
overhead and a clearer path toward real-time deployment.

The implementation is intentionally direct: classes, Eigen matrices, MuJoCo C
API calls, and OSQP C API calls. It avoids ROS 2 and template-heavy framework
code so the control math remains visible.

## Implemented

```text
MuJoCo Go2 model interface
M(q), h(q,v) = qfrc_bias - qfrc_passive
cached actuation matrix B
foot positions and foot Jacobians
whole-body WBC QP:
    z = [vdot, tau, contact_force]
    M vdot + h = B tau + Jc^T f
    stance no-slip acceleration constraint
    swing foot acceleration tracking cost
    friction pyramid
    torque limits
SRB / centroidal MPC QP:
    state = [com, com_velocity, rpy, omega]
    input = [f_FL, f_FR, f_RL, f_RR]
    per-knot contact schedule
    swing foot force = 0 hard constraint
headless diagonal trot rollout:
    MPC force reference -> WBC torque -> MuJoCo step
```

This is a real QP-based controller path. The rollout applications do not use a
PD-only fallback or a pre-recorded torque sequence.

## Build On Windows

The repository can be built with a portable MinGW toolchain. On this machine the
toolchain was extracted outside the repository:

```powershell
D:\projects\quadruped_project\tools\w64devkit
```

From the repository root:

```powershell
$env:Path = "D:\projects\quadruped_project\tools\w64devkit\bin;" + $env:Path
.\..\tools\w64devkit\bin\cmake.exe -S cpp -B cpp\build-osqp -G Ninja "-DCMAKE_CXX_COMPILER=g++.exe" "-DCMAKE_MAKE_PROGRAM=D:\projects\quadruped_project\tools\w64devkit\bin\ninja.exe"
.\..\tools\w64devkit\bin\cmake.exe --build cpp\build-osqp --config Release
```

CMake downloads Eigen 3.4.0 and OSQP v1.0.0 if they are not installed. The
MuJoCo Python wheel is used for headers and `mujoco.dll`; CMake generates a
MinGW import library for the DLL when needed.

## Run

From the repository root:

```powershell
.\cpp\build-osqp\inspect_dynamics.exe .\models\mujoco_menagerie\unitree_go2\scene.xml
.\cpp\build-osqp\solve_wbc_once.exe .\models\mujoco_menagerie\unitree_go2\scene.xml
.\cpp\build-osqp\solve_mpc_once.exe .\models\mujoco_menagerie\unitree_go2\scene.xml
.\cpp\build-osqp\run_trot_rollout.exe .\models\mujoco_menagerie\unitree_go2\scene.xml 0.012 0.08
.\cpp\build-osqp\run_trot_rollout.exe .\models\mujoco_menagerie\unitree_go2\scene.xml route cpp_outputs/cpp_trot_route.csv
```

The rollout arguments are:

```text
run_trot_rollout.exe <model.xml> <vx_mps> <yaw_rate_radps> [record_csv]
run_trot_rollout.exe <model.xml> route [record_csv]
```

The `route` mode is a simple sequence: straight walking, left turning, more
straight walking, and a final stop. It is intended for performance and
architecture checks. The Python route demo currently has more mature
reference/planner tuning and should be used for public visual presentation.

To save a C++ route rollout and render it as a debugging GIF:

```powershell
.\cpp\build-osqp\run_trot_rollout.exe .\models\mujoco_menagerie\unitree_go2\scene.xml route cpp_outputs/cpp_trot_route.csv
.\.venv\Scripts\python.exe -B .\scripts\render_cpp_rollout_gif.py --csv .\cpp_outputs\cpp_trot_route.csv --gif-output .\cpp_outputs\cpp_trot_route.gif --stride 4 --fps 30 --playback-speed 2.0
```

To view the same C++ trajectory in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\render_cpp_rollout_gif.py --csv .\cpp_outputs\cpp_trot_route.csv --no-gif --viewer-replay --stride 2
```

## Local Benchmark Example

On the current Windows setup, the headless trot rollout produced:

```text
straight / turning trot: sim_per_wall ~= 1.8-2.2
average WBC solve ~= 0.36-0.40 ms
average MuJoCo step ~= 0.04 ms
average MPC solve ~= 31-37 ms
```

This shows the main Python bottleneck was not MuJoCo dynamics itself. The C++
WBC path is already fast; the current MPC still spends time in a fresh dense
assembly plus OSQP solve and can be optimized further by fixing sparsity,
preallocating matrices, and updating only numeric values.
