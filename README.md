# MuJoCo Go2 MPC/WBC Locomotion Study

This repository is a study and research prototype on model-based quadruped
locomotion control. I built it to understand how a floating-base legged robot
can be controlled through a layered MPC/WBC architecture in MuJoCo.

The current implementation is not meant to be an industrial locomotion stack.
Its purpose is to make the main mathematical interfaces explicit and test them
in simulation:

```text
MuJoCo floating-base state
    -> reference and contact schedule
    -> single-rigid-body MPC
    -> full-body WBC QP
    -> joint torque command
    -> MuJoCo simulation step
```

![Trot route demo](docs/assets/trot_l_route_demo.gif)

## Motivation

The project started from a simple question:

```text
How can a quadruped controller go beyond foot-level IK or joint-space PD,
and instead compute torques from the full floating-base dynamics?
```

The implementation therefore focuses on:

- MuJoCo model access for a Unitree Go2 robot;
- floating-base generalized coordinates and velocities;
- mass matrix, bias force, actuation matrix, foot Jacobians, and `Jdot*v`;
- contact-force planning with a single-rigid-body MPC;
- torque generation with a full-body WBC quadratic program;
- stance, swing, crawl-like, and diagonal-trot contact modes in simulation.

## Current Status

The repository contains two versions of the control stack.

The Python version is the main experimental version. It is easier to inspect and
contains the reference and demonstration scripts used in the current examples.

The C++ version is a smaller implementation using Eigen, the MuJoCo C API, and
the OSQP C API. It was added to check the same MPC/WBC formulation in a more
real-time-oriented language, but it is still a research implementation rather
than a tuned controller.

Implemented so far:

```text
Unitree Go2 MuJoCo model interface
SRB-MPC contact-force QP
full-body WBC-QP with z = [vdot, tau, f]
four-foot stance
single-leg swing
crawl-like contact modes
diagonal trot contact modes
offline trot route replay
basic WBC profiling
C++ headless MPC/WBC rollout
```

Current limitations:

```text
no real robot deployment
no hardware state estimator
limited touchdown/load-transfer tuning
limited gait and foothold planning
not optimized as a real-time control stack
no terrain locomotion or large-scale RL training yet
```

## Demo

The clearest visual demo is an offline rollout followed by fixed-rate replay in
the MuJoCo viewer:

```powershell
cd D:\projects\quadruped_project\mujoco_wbc_project
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-turn-stop --no-gif --viewer-replay
```

The route contains:

```text
straight walking
left turn of approximately 90 degrees
short recovery pause
additional straight walking
final stop
```

A shorter straight-walking replay can be run with:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset straight --no-gif --viewer-replay
```

To regenerate the GIF used above:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-turn-stop
```

For controller debugging, the live viewer runs MPC/WBC online while drawing:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\run_trot_reference_viewer.py --vx 0.012
```

The live viewer can look less smooth than replay because Python solves the QPs
while rendering.

## Setup

The project was developed with Python 3.12 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Unitree Go2 model is included through the `mujoco_menagerie` submodule:

```powershell
git submodule update --init --recursive
```

Basic checks:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\check_mujoco_install.py
.\.venv\Scripts\python.exe -B .\scripts\launch_go2_viewer.py
.\.venv\Scripts\python.exe -B .\scripts\validate_control_stack.py
```

## Mathematical Interface

### MuJoCo State Convention

The Go2 model uses MuJoCo's floating-base representation:

```text
qpos[0:3] = base position p_WB, expressed in world frame W
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity, expressed in W
qvel[3:6] = base angular velocity, expressed in W
qvel[6:]  = 12 joint velocities
```

Therefore:

```text
nq = 19
nv = 18
nu = 12
```

### WBC-QP

The WBC decision variable is:

```text
z = [vdot, tau, f]
```

where:

```text
vdot in R^18     generalized acceleration
tau  in R^12     joint torque command
f    in R^(3nc)  stance-foot contact forces in world frame
```

The main dynamics constraint is:

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
```

The WBC also uses:

```text
stance-foot acceleration constraints
friction pyramid constraints
torque limits
swing-foot acceleration tracking
base/body reference tracking
MPC contact-force reference tracking
regularization on torque and contact force
```

### SRB-MPC

The MPC uses a single-rigid-body approximation:

```text
x = [com_pos, com_vel, theta, omega]
u = [f_FL, f_FR, f_RL, f_RR]
```

with discrete dynamics:

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / m + g)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]
```

The MPC predicts per-foot contact forces. The WBC then tracks the first-knot
force reference while enforcing the full floating-base dynamics at the current
simulation step.

## Repository Layout

```text
src/mujoco_wbc/
    model_interface.py       MuJoCo dynamics and kinematics wrapper
    centroidal_mpc.py        SRB-MPC contact-force QP
    wbc_qp.py                full-body WBC QP
    contact_schedule.py      stance/swing schedule helpers
    swing_trajectory.py      swing-foot reference generation
    planning.py              simple command/reference helpers

scripts/
    validate_control_stack.py
    record_trot_demo.py
    run_trot_reference_viewer.py
    run_commanded_crawl_viewer.py
    run_static_stance_viewer.py
    run_single_leg_swing_viewer.py
    inspect_go2_dynamics.py
    inspect_frame_conventions.py

cpp/
    include/go2wbc/          C++ interfaces
    src/                     MuJoCo interface, OSQP wrapper, MPC, WBC
    apps/                    small executables for checks and rollout

docs/
    control_stack.md
    mainline_architecture.md
    project_structure.md
    locomotion_reference_map.md
```

## Possible Tesina Direction

One possible continuation of this work is:

```text
Learning-Based Foothold and Reference Adaptation
for MPC/WBC-Based Quadruped Locomotion
```

The idea would be to keep MPC/WBC as the model-based low-level controller and
add a lightweight learning-based module that adapts foothold placement or
body-reference commands. A small tesina-scale version could stay fully in
MuJoCo simulation and compare the learned adapter with the current rule-based
reference generation during diagonal trot, turning, and mild disturbances.

Other possible continuations:

```text
ROS 2 / C++ integration
fixed-size real-time QP formulation
contact transition and touchdown handling
more systematic gait and foothold planning
terrain-aware locomotion experiments
```
