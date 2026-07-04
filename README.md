# MuJoCo Go2 SRB-MPC and Full-Body WBC Locomotion

This repository implements a dynamics-based quadruped locomotion control stack
for a Unitree Go2 model in MuJoCo. The project focuses on a clear and
inspectable implementation of single-rigid-body MPC and full-body whole-body
control, rather than on highly tuned high-speed gait performance.

The main control pipeline is:

```text
MuJoCo floating-base state
    -> reference and contact schedule
    -> single-rigid-body MPC
    -> full-body WBC QP
    -> joint torque command
    -> MuJoCo simulation step
```

![Trot route demo](docs/assets/trot_l_route_demo.gif)

## Highlights

- Unitree Go2 floating-base MuJoCo model with `nq=19`, `nv=18`, and `nu=12`.
- Explicit generalized coordinate, generalized velocity, world-frame, base-frame,
  and foot Jacobian conventions.
- MuJoCo-based access to `M(q)`, `h(q,v)`, `B`, foot Jacobians, `Jdot*v`, COM,
  and composite inertia quantities.
- Single-rigid-body / centroidal MPC that optimizes per-foot world-frame contact
  force references over a prediction horizon.
- Full-body WBC QP with decision variables `[vdot, tau, f]`.
- Support for four-foot stance, single-leg swing, crawl-like contact modes, and
  diagonal trot contact modes through a generic contact-mode WBC.
- Offline rollout and fixed-rate MuJoCo replay for smooth demonstration videos,
  independent of Python QP runtime.
- Basic profiling hooks for the WBC solve path and cached actuation matrix
  support for the Go2 MuJoCo model.

## Demo

Recommended public demo:

```powershell
cd D:\projects\quadruped_project\mujoco_wbc_project
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-turn-stop --no-gif --viewer-replay
```

This command first rolls out the closed-loop controller headlessly, stores the
state trajectory, and then replays the stored trajectory in the MuJoCo viewer at
a fixed visual frame rate.

The `trot-l-turn-stop` preset performs:

```text
straight walking
left turn of approximately 90 degrees
short recovery pause
additional straight walking
final stop
```

Short straight-walking replay:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset straight --no-gif --viewer-replay
```

Generate a GIF:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-turn-stop
```

Live viewer version for debugging:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\run_trot_reference_viewer.py --vx 0.012
```

The live viewer solves MPC/WBC while drawing frames, so it can look less smooth
than the offline replay on slower Python runs.

## Installation

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The Unitree Go2 model is provided through the `mujoco_menagerie` Git submodule.
After cloning the repository, initialize submodules:

```powershell
git submodule update --init --recursive
```

Check the MuJoCo installation:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\check_mujoco_install.py
```

Open the raw Go2 model:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\launch_go2_viewer.py
```

## Validation

Run the control-stack regression checks:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\validate_control_stack.py
```

The validation covers:

```text
actuation matrix cache consistency
contact phase semantics
static stance WBC
single-leg swing WBC
general WBC crawl mode
general WBC diagonal trot mode
SRB-MPC all-stance mode
SRB-MPC swing-foot force-zero constraint
```

Low-frame-rate full route stability check:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-route --no-gif --fps 1 --log-dt 20 --stop-on-fall
```

WBC internal timing in the live viewer:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\run_trot_reference_viewer.py --vx 0.012 --wbc-profile
```

## Project Layout

```text
mujoco_wbc_project/
|-- README.md
|-- requirements.txt
|-- models/
|   |-- free_body_smoke.xml
|   `-- mujoco_menagerie/        # Git submodule with the Unitree Go2 model
|-- src/mujoco_wbc/
|   |-- model_interface.py       # MuJoCo dynamics and kinematics wrapper
|   |-- centroidal_mpc.py        # SRB-MPC contact-force QP
|   |-- wbc_qp.py                # full-body WBC QP
|   |-- contact_schedule.py      # stance/swing schedule helpers
|   |-- swing_trajectory.py      # swing-foot reference generation
|   |-- planning.py              # simple command/reference helpers
|   |-- reference_inputs.py      # reference and contact-mode data structures
|   |-- support_polygon.py       # support geometry helpers
|   |-- profiling.py             # loop timing utilities
|   `-- conventions.py           # coordinate and naming conventions
|-- scripts/
|   |-- validate_control_stack.py
|   |-- record_trot_demo.py
|   |-- run_trot_reference_viewer.py
|   |-- run_commanded_crawl_viewer.py
|   |-- run_srb_mpc_crawl_continuous_viewer.py
|   |-- run_static_stance_once.py
|   |-- run_static_stance_viewer.py
|   |-- run_single_leg_swing_once.py
|   |-- run_single_leg_swing_viewer.py
|   |-- inspect_go2_dynamics.py
|   |-- inspect_frame_conventions.py
|   |-- launch_go2_viewer.py
|   `-- check_mujoco_install.py
`-- docs/
    |-- control_stack.md
    |-- mainline_architecture.md
    |-- project_structure.md
    `-- locomotion_reference_map.md
```

## Mathematical Interface

### Generalized Coordinates

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

### Full-Body WBC QP

The WBC decision variable is:

```text
z = [vdot, tau, f]
```

with:

```text
vdot in R^18     generalized acceleration
tau  in R^12     joint torque command
f    in R^(3nc)  stance-foot contact forces, expressed in world frame
```

Hard constraints:

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
J_c(q) vdot + Jdot_c(q,v) v = a_c_cmd
tau_min <= tau <= tau_max
|fx| <= mu fz
|fy| <= mu fz
fz >= 0
```

Soft tasks:

```text
base position and orientation acceleration tracking
nominal joint posture tracking
swing-foot acceleration tracking
MPC contact-force reference tracking
torque and contact-force regularization
```

### SRB-MPC

The MPC uses a single-rigid-body / centroidal approximation:

```text
x = [com_pos, com_vel, theta, omega]
u = [f_FL, f_FR, f_RL, f_RR]
```

Discrete dynamics:

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / m + g)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]
```

`theta` is a small-angle orientation state around the current linearization
point, not a global quaternion state.

Contact constraints:

```text
stance foot: friction pyramid, fz >= normal_force_min
swing foot:  f = 0
```

The MPC outputs the first-knot per-foot contact force reference. The WBC then
tracks that force reference while enforcing the full floating-base dynamics.

## Current Scope

Implemented and validated:

```text
MuJoCo Go2 model interface
SRB-MPC contact-force planning
full-body WBC QP
generic non-flight contact-mode WBC
stance, single-leg swing, crawl-mode, and diagonal-trot checks
stable offline route replay demo
WBC profiling and cached actuation matrix validation
```

Out of scope for the current Python prototype:

```text
real-time C++ implementation
hardware state estimation
hardware deployment
robust touchdown and load-transfer handling
high-speed natural trot
terrain locomotion
large-scale RL training
```

## Future Work

Possible extensions include:

```text
1. C++/ROS 2 implementation with fixed-size data structures and reusable QP memory.
2. Fixed-sparsity WBC formulation with a constant 42-dimensional decision vector.
3. Improved foothold and body-reference generation.
4. Contact transition handling with force ramps and touchdown hysteresis.
5. Hardware-oriented state-estimation and actuator-interface integration.
6. RL residual policies on top of MPC/WBC references.
```
