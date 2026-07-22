# Project Structure

The repository separates reusable control code, executable scripts, model
assets, and documentation:

```text
src/mujoco_wbc/  reusable control library
scripts/         executable checks, demos, and viewers
models/          MuJoCo XML and mesh assets
docs/            architecture and mathematical interface notes
```

## 1. Core Library

```text
src/mujoco_wbc/model_interface.py
```

MuJoCo dynamics and kinematics wrapper. It exposes:

```text
q, v
M(q)
h(q,v)
B
J_foot
Jdot_foot * v
COM
base rotation
foot contact positions
```

This file is the boundary between MuJoCo simulation data and the controller
mathematical interface.

```text
src/mujoco_wbc/centroidal_mpc.py
```

Single-rigid-body MPC. It optimizes predicted COM states and per-foot contact
forces over a finite horizon.

```text
src/mujoco_wbc/wbc_qp.py
```

Full-body WBC QP. The decision variable is:

```text
z = [vdot, tau, f]
```

The QP computes joint torque commands while enforcing floating-base dynamics,
stance constraints, torque limits, and friction constraints.

```text
src/mujoco_wbc/contact_schedule.py
```

Contact-phase helpers. These functions express which feet are in stance and
which feet are in swing over the current time and MPC horizon.

```text
src/mujoco_wbc/swing_trajectory.py
```

Swing-foot trajectory generation. Given a start point, target point, swing
height, and timing, it returns position, velocity, and acceleration references.

```text
src/mujoco_wbc/planning.py
```

Minimal command and reference helpers. This file provides a simple path from
velocity commands to foothold and body references. It is intentionally not a
production gait planner.

```text
src/mujoco_wbc/reference_inputs.py
src/mujoco_wbc/support_polygon.py
src/mujoco_wbc/profiling.py
src/mujoco_wbc/conventions.py
```

Supporting modules for reference validation, support geometry, timing, and
coordinate/name conventions.

## 2. Executable Scripts

Environment and model checks:

```text
scripts/check_mujoco_install.py
scripts/launch_go2_viewer.py
scripts/inspect_go2_dynamics.py
scripts/inspect_frame_conventions.py
```

Control-stack regression:

```text
scripts/validate_control_stack.py
```

Basic WBC demos:

```text
scripts/run_static_stance_once.py
scripts/run_static_stance_viewer.py
scripts/run_single_leg_swing_once.py
scripts/run_single_leg_swing_viewer.py
```

Main locomotion demos:

```text
scripts/record_trot_demo.py
scripts/run_trot_reference_viewer.py
scripts/run_commanded_crawl_viewer.py
```

`record_trot_demo.py` first performs a headless closed-loop rollout, then
replays the stored states at a fixed visual frame rate.

## 3. Recommended Review Order

For understanding the implementation:

1. `README.md`
2. `docs/control_stack.md`
3. `docs/mainline_architecture.md`
4. `scripts/validate_control_stack.py`
5. `src/mujoco_wbc/model_interface.py`
6. `src/mujoco_wbc/wbc_qp.py`
7. `src/mujoco_wbc/centroidal_mpc.py`
8. `src/mujoco_wbc/swing_trajectory.py`
9. `scripts/run_trot_reference_viewer.py`
10. `scripts/record_trot_demo.py`

The important questions for each module are:

```text
What is the state?
What is the input?
What is the decision variable?
What is the cost?
What is the constraint?
Who consumes the output?
```

Module-level mapping:

```text
model_interface.py
  input: MuJoCo model/data
  output: dynamics and kinematics quantities

centroidal_mpc.py
  input: COM reference, contact schedule, current robot state
  decision: horizon states and foot forces
  output: first-knot per-foot force reference

wbc_qp.py
  input: full dynamics, contact mode, swing/base references, MPC force reference
  decision: [vdot, tau, f]
  output: joint torque tau

viewer/demo scripts
  input: command/reference parameters
  output: MuJoCo simulation, replay, or rendered GIF
```
