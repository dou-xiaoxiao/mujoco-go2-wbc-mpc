# Project Structure and Reading Guide

This document is a map for reading the project. The short version is:

```text
scripts/         runnable experiments and demos
src/mujoco_wbc/  reusable control library
models/          MuJoCo XML and assets
docs/            explanations and project notes
```

`scripts/` is where you start a program. `src/mujoco_wbc/` is where the actual
controller code lives.

## 1. Top-Level Files

```text
README.md
```

Project entry point: setup, model choice, and common run commands.

```text
requirements.txt
```

Python dependencies for the virtual environment.

```text
.gitignore
```

Tells Git to ignore generated files such as `.venv/`, `__pycache__/`, and logs.

```text
.gitmodules
```

Declares `models/mujoco_menagerie` as a Git submodule.

```text
models/free_body_smoke.xml
```

Tiny MuJoCo model used to verify MuJoCo APIs.

```text
models/mujoco_menagerie/
```

External model collection. The Unitree Go2 XML and assets come from here.

## 2. Core Library: `src/mujoco_wbc`

```text
src/mujoco_wbc/__init__.py
```

Package facade. It lets scripts write:

```python
from mujoco_wbc import Go2ModelInterface, CentroidalMPC
```

instead of importing from every internal file manually.

```text
src/mujoco_wbc/conventions.py
```

Shared naming and frame conventions: foot names, body names, and base/world
coordinate assumptions.

```text
src/mujoco_wbc/model_interface.py
```

The MuJoCo wrapper. It reads the robot state and exposes dynamics quantities:

```text
q, v
M(q)
h(q, v)
B
foot Jacobians
Jdot * v
COM position / velocity
base rotation matrix
contact positions
```

This file is the bridge between MuJoCo and our controller math.

```text
src/mujoco_wbc/wbc_qp.py
```

Full-body WBC QP. This is the main lower-level controller.

Decision variable:

```text
z = [vdot, tau, f]
```

Main hard constraint:

```text
M(q) vdot + h(q, v) = B tau + J_c(q)^T f
```

It also handles stance constraints, swing tracking tasks, torque limits,
friction pyramids, and MPC force tracking.

Important classes:

```text
StanceWBCQP
SingleLegSwingWBCQP
GeneralContactWBCQP
```

`GeneralContactWBCQP` is the more general one used for crawl/trot-style contact
modes.

```text
src/mujoco_wbc/centroidal_mpc.py
```

SRB/centroidal MPC. It optimizes COM/body motion and foot contact forces over a
horizon.

State:

```text
x = [com_pos, com_vel, theta, omega]
```

Input:

```text
u = [f_FL, f_FR, f_RL, f_RR]
```

The MPC output is a per-foot force reference. WBC then tries to realize that
force using the full floating-base dynamics.

```text
src/mujoco_wbc/contact_schedule.py
```

Contact-mode helpers: which foot is stance, which foot is swing, and how that
changes over time.

```text
src/mujoco_wbc/swing_trajectory.py
```

Swing foot trajectory generation: start point, target foothold, lift height,
smooth position/velocity/acceleration references.

```text
src/mujoco_wbc/planning.py
```

Simple command-to-reference layer. It turns velocity commands into foothold and
body/COM references for crawl-style walking.

```text
src/mujoco_wbc/reference_inputs.py
```

Common data structures for references and contact modes. This is the interface
between an upper planner and the MPC/WBC stack.

```text
src/mujoco_wbc/support_polygon.py
```

Geometry helpers for support sets. Useful for thinking about whether the body
reference is reasonable relative to stance feet.

```text
src/mujoco_wbc/profiling.py
```

Small timing utility. It measures how much wall time is spent in MuJoCo, MPC,
WBC, viewer sync, and diagnostics.

## 3. Scripts

Scripts are executable entry points. Most scripts start with a small block that
adds `src/` to Python's import path, then imports the reusable code.

### First Scripts To Read

```text
scripts/launch_go2_viewer.py
```

Open the Go2 MuJoCo model.

```text
scripts/validate_control_stack.py
```

Fast sanity check for the full stack. This should stay green when refactoring.

```text
scripts/inspect_go2_dynamics.py
```

Prints model dimensions and dynamics quantities. Good for understanding
generalized coordinates and MuJoCo data.

```text
scripts/inspect_frame_conventions.py
```

Checks base/world frame conventions, rotation matrices, and foot positions.

### WBC Learning Scripts

```text
scripts/run_static_stance_once.py
scripts/run_static_stance_viewer.py
```

Four-foot stance WBC.

```text
scripts/run_single_leg_swing_once.py
scripts/run_single_leg_swing_viewer.py
scripts/run_single_leg_forward_step_viewer.py
```

Single-leg swing WBC: one foot is removed from stance and tracked as a swing
task.

### MPC Learning Scripts

```text
scripts/run_centroidal_mpc_once.py
scripts/run_centroidal_mpc_horizon_once.py
```

MPC-only checks.

```text
scripts/run_centroidal_to_wbc_once.py
scripts/run_horizon_mpc_to_wbc_once.py
```

MPC-to-WBC bridge checks.

### Locomotion Viewer Scripts

```text
scripts/run_commanded_crawl_viewer.py
```

Commanded crawl demo.

```text
scripts/run_trot_reference_viewer.py
```

Current trot reference demo. This is the main dynamic-gait experiment.

```text
scripts/debug_trot_headless.py
```

No-viewer trot debug script. Use this for logs and speed profiling.

### Older Milestone Scripts

These are still useful as development history, but they are not the first files
to read:

```text
scripts/run_srb_mpc_crawl_viewer.py
scripts/run_srb_mpc_crawl_forward_viewer.py
scripts/run_srb_mpc_crawl_continuous_viewer.py
scripts/run_srb_mpc_forward_step_viewer.py
scripts/run_step_sequence_viewer.py
scripts/simulate_*.py
scripts/inspect_contact_schedule.py
scripts/inspect_crawl_planner.py
scripts/inspect_external_reference_modes.py
```

## 4. Best Reading Order

1. `README.md`
2. `docs/mainline_architecture.md`
3. `scripts/validate_control_stack.py`
4. `src/mujoco_wbc/model_interface.py`
5. `scripts/inspect_go2_dynamics.py`
6. `scripts/inspect_frame_conventions.py`
7. `src/mujoco_wbc/wbc_qp.py`
8. `src/mujoco_wbc/centroidal_mpc.py`
9. `src/mujoco_wbc/swing_trajectory.py`
10. `src/mujoco_wbc/reference_inputs.py`
11. `scripts/run_trot_reference_viewer.py`
12. `scripts/debug_trot_headless.py`

The key is to read from signal source to actuator output:

```text
MuJoCo state
  -> model interface
  -> MPC references and force plan
  -> WBC QP
  -> tau
  -> MuJoCo step
```

## 5. What `import` Means Here

`import` is not text copy. It means:

1. Python finds a `.py` file.
2. Python runs that file once to create a module object.
3. Other files can use classes/functions/variables from that module.

Example:

```python
from mujoco_wbc import Go2ModelInterface
```

This does not paste `model_interface.py` into the script. It makes the class
`Go2ModelInterface` available by name.

Most scripts also contain something like:

```python
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
```

That tells Python:

```text
also search this project's src/ directory when importing packages
```

This is why scripts can import `mujoco_wbc` without installing the package.

This pattern:

```python
if __name__ == "__main__":
    main()
```

means:

```text
run main() only when this file is executed as a script
do not run main() just because another file imported it
```

## 6. Python Structures Used Most

### Dictionary

A dictionary is a key-value map:

```python
foot_positions = {
    "FL": np.array([0.2, 0.1, 0.0]),
    "FR": np.array([0.2, -0.1, 0.0]),
}
```

Read one value:

```python
fl_pos = foot_positions["FL"]
```

Loop over keys and values:

```python
for foot, pos in foot_positions.items():
    print(foot, pos)
```

Safe lookup with a default:

```python
value = foot_positions.get("RL", default_value)
```

In this project, dictionaries are often used for per-foot data:

```text
foot -> position
foot -> force reference
foot -> swing plan
foot -> contact state
```

Common pattern:

```python
initial_foot_positions = {
    foot: robot.geom_position(foot)
    for foot in FOOT_GEOMS
}
```

This is a dictionary comprehension. It means:

```text
for every foot name in FOOT_GEOMS, compute and store its position
```

### Tuple

A tuple is an immutable ordered group:

```python
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
```

Tuples are used for fixed orders and dictionary keys:

```python
key = (tuple(stance_feet), tuple(swing_feet))
controller = controller_cache[key]
```

Lists can change; tuples usually mean "this order should stay fixed."

### Dataclass

A dataclass is a compact way to define a structured object:

```python
@dataclass(frozen=True)
class GaitConfig:
    step_time: float
    swing_height: float
```

Create one:

```python
cfg = GaitConfig(step_time=0.4, swing_height=0.05)
```

Read fields:

```python
cfg.step_time
cfg.swing_height
```

In this project, dataclasses are mostly configs, references, windows, and solver
results.

### NumPy Array

Most math is in `np.ndarray`:

```python
x = np.array([1.0, 2.0, 3.0])
```

Vectors and matrices are sliced often:

```python
base_pos = qpos[0:3]
base_quat = qpos[3:7]
joint_pos = qpos[7:]
```

This means:

```text
qpos[0:3]  -> elements 0, 1, 2
qpos[3:7]  -> elements 3, 4, 5, 6
qpos[7:]   -> element 7 to the end
```

### Class and `self`

A class groups data and functions:

```python
mpc = CentroidalMPC(config)
result = mpc.solve(...)
```

Inside the class, `self` means "this specific object":

```python
self.config
self._solver
```

So `self._solver` is the OSQP solver owned by that MPC object.

### Type Hints

This:

```python
force_refs: dict[str, np.ndarray]
```

means:

```text
force_refs is expected to be a dictionary
keys are strings
values are NumPy arrays
```

Type hints mostly help humans and editors. They usually do not change runtime
behavior.

## 7. The Mental Model

When reading any file, ask:

```text
state 是什么？
input 是什么？
decision variable 是什么？
cost 是什么？
constraint 是什么？
output 给谁？
```

For this project:

```text
model_interface.py
  input: MuJoCo data
  output: dynamics quantities

centroidal_mpc.py
  input: current COM state, contact schedule, references
  decision: horizon states and foot forces
  output: current per-foot force reference

wbc_qp.py
  input: MuJoCo dynamics, contact mode, swing/base refs, MPC force refs
  decision: [vdot, tau, f]
  output: joint torque tau

viewer scripts
  input: command/reference parameters
  output: running simulation
```

This is the core structure to understand before tuning trot behavior.
