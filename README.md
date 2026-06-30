# MuJoCo WBC/MPC Quadruped Project

This is the new simulator-agnostic WBC/MPC track. It starts with MuJoCo
because MuJoCo exposes generalized-coordinate dynamics and Jacobians directly.

## Environment

Virtual environment:

```text
D:\projects\quadruped_project\mujoco_wbc_project\.venv
```

Run Python:

```powershell
D:\projects\quadruped_project\mujoco_wbc_project\.venv\Scripts\python.exe
```

Installed core packages:

```text
mujoco
numpy
scipy
osqp
matplotlib
```

## First Goal

Build a clean model interface for WBC:

```text
q, v
M(q)
h(q, v)
frame Jacobian J
Jdot v
contact constraints
torque limits
```

Then solve:

```text
M(q) vdot + h(q, v) = S^T tau + J_c(q)^T f
```

with WBC decision variables:

```text
vdot, tau, f
```

## Current Scripts

```text
scripts/check_mujoco_install.py
scripts/inspect_go2_dynamics.py
scripts/inspect_frame_conventions.py
scripts/run_static_stance_once.py
scripts/simulate_static_stance.py
scripts/run_static_stance_viewer.py
scripts/run_centroidal_mpc_once.py
scripts/run_centroidal_mpc_horizon_once.py
scripts/run_centroidal_to_wbc_once.py
scripts/run_horizon_mpc_to_wbc_once.py
scripts/inspect_contact_schedule.py
scripts/simulate_srb_mpc_forward_step.py
scripts/run_srb_mpc_forward_step_viewer.py
scripts/simulate_srb_mpc_crawl.py
scripts/run_single_leg_swing_once.py
scripts/simulate_single_leg_swing.py
scripts/run_single_leg_swing_viewer.py
scripts/simulate_single_leg_forward_step.py
scripts/run_single_leg_forward_step_viewer.py
scripts/simulate_step_sequence.py
scripts/run_step_sequence_viewer.py
```

Verifies that MuJoCo imports and exposes mass matrix, Jacobian, and inverse
dynamics calls.

## Current Stability Status

Stable baseline:

```text
1. Static four-foot stance WBC
2. Three-stance plus FL swing WBC
3. FL forward-step WBC with touchdown and re-stance
4. SRB-MPC force reference connected into the FL forward-step WBC loop
```

Experimental:

```text
1. FL -> RR -> FR -> RL crawl contact sequence
2. Four-leg step sequence
```

Those experimental scripts are useful for exposing the missing upper-layer
body-reference and foothold scheduling problems, but they should not yet be
treated as completed gaits.

Control stack notes:

```text
docs/mainline_architecture.md
docs/control_stack.md
docs/locomotion_reference_map.md
docs/external_reference_sources.md
```

Quick regression check for the current MPC/WBC interfaces:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_control_stack.py
```

Inspect which upstream gait/contact references the current WBC can execute:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_external_reference_modes.py
```

Run the continuous crawl in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_commanded_crawl_viewer.py
```

The viewer demo exposes the upper-layer command directly:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_commanded_crawl_viewer.py --demo forward
.\.venv\Scripts\python.exe .\scripts\run_commanded_crawl_viewer.py --vx 0.003 --vy 0.0 --yaw-rate 0.0
.\.venv\Scripts\python.exe .\scripts\run_commanded_crawl_viewer.py --demo stand-step
```

Run a diagonal-pair trot reference through the generic contact-mode WBC:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_trot_reference_viewer.py --vx 0.012
.\.venv\Scripts\python.exe .\scripts\run_trot_reference_viewer.py --vx 0.008 --yaw-rate 0.04
```

For a faster viewer demo, lower solve/display rates or skip the finite-difference
`Jdot*v` term:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_trot_reference_viewer.py --vx 0.012 --viewer-hz 30 --wbc-dt 0.012 --mpc-dt 0.06
.\.venv\Scripts\python.exe .\scripts\run_trot_reference_viewer.py --vx 0.012 --ignore-jdot-v
```

The continuous crawl scripts are command-driven:

```text
CrawlCommand(vx, vy, yaw_rate)
    -> command-based foothold deltas
    -> support-centroid body reference
    -> COM position/velocity reference
    -> centroidal MPC force reference
    -> WBC torque command
```

The default viewer command is deliberately conservative:

```text
vx = 0.003 m/s
vy = 0.000 m/s
yaw_rate = 0.000 rad/s
```

With the default crawl timing, this produces:

```text
step_delta = [0.024, 0.0, 0.0] m
```

The viewer loop uses separate update rates:

```text
MuJoCo physics timestep: model.opt.timestep
WBC_UPDATE_DT:           torque QP update period
VIEWER_SYNC_DT:          viewer refresh period
MPC_UPDATE_DT:           centroidal MPC update period
PROFILE_LOG_DT:          timing summary print period
```

`LoopProfiler` prints compact wall-clock timing summaries and real-time factor:

```text
profile: sim=...s wall=...s rtf=... | schedule: n=... mean=...ms max=...ms | mpc: n=... | wbc: n=... | mj_step: n=... | viewer: n=...
```

This tells us whether viewer slowness is coming from MPC, WBC, MuJoCo stepping,
viewer sync, or intentional sleep. Physics still steps at the MuJoCo timestep;
between WBC updates the viewer holds the last solved torque command.

QP solver reuse status:

```text
CentroidalMPC:  first call OSQP setup, later calls update q/l/u/Ax and warm start
StanceWBCQP:    first call OSQP setup, later calls update q/l/u/Ax and warm start
SwingWBCQP:     first call OSQP setup, later calls update P/q/l/u/Ax and warm start
```

In QP form, `P` is the Hessian of the quadratic objective, not just a raw
weight. For example, a swing tracking term `w ||J_sw vdot - target||^2`
contributes `w J_sw^T J_sw` to `P` and `-w J_sw^T target` to `q`.

The remaining WBC speed bottleneck is now mostly matrix/Jacobian construction
before the OSQP call, not only solver setup.

## Go2 Dynamics Milestone

Robot model:

```text
models/mujoco_menagerie/unitree_go2/scene.xml
```

The Unitree Go2 model is loaded as a floating-base quadruped:

```text
nq = 19
nv = 18
nu = 12
```

MuJoCo convention:

```text
qpos[0:3] = base xyz
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:] = 12 joint positions

qvel[0:3] = base linear velocity
qvel[3:6] = base angular velocity
qvel[6:] = 12 joint velocities
```

Coordinate and frame convention:

```text
W = world frame
B = floating base body frame
F = foot point/frame

R_WB maps vectors from B to W
R_BW = R_WB.T maps vectors from W to B

p_WB = base position in world
p_WF = foot point position in world
p_BF = R_BW (p_WF - p_WB)
```

MuJoCo Jacobian convention used by WBC:

```text
v_WF = J_WF(q) v
xddot_WF = J_WF(q) vdot + Jdot_WF(q, v) v
```

Contact force convention:

```text
f_i = [fx, fy, fz] expressed in W at the foot point
generalized contact force = J_WF(q)^T f_i
```

Inspect frame conversions and foot coordinates:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_frame_conventions.py
```

The first WBC dynamics interface exposes:

```text
M(q) vdot + h(q, v) = B tau + Jc(q)^T f
```

with:

```text
M  in R^(18 x 18)
h  in R^18
B  in R^(18 x 12)
Jc in R^(12 x 18) for four 3D foot contact forces
f  in R^12
```

Inspect the model and dynamics blocks:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_go2_dynamics.py
```

Solve one static four-foot stance WBC QP:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_static_stance_once.py
```

Run a short headless stance torque-control smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_static_stance.py
```

Run the stance WBC in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_static_stance_viewer.py
```

Solve one centroidal contact-force QP:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_centroidal_mpc_once.py
```

The first centroidal layer solves:

```text
decision variable = [f_FL, f_FR, f_RL, f_RR]

sum_i f_i = m (a_com_des - g)
sum_i (p_i - p_com) x f_i = tau_com_des
```

with friction-pyramid constraints. This is currently a single-node force QP,
not a full horizon MPC yet; the purpose is to establish the centroidal wrench
interface that provides contact force references to WBC.

Verify the first MPC-to-WBC data path:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_centroidal_to_wbc_once.py
```

Solve the first horizon centroidal MPC:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_centroidal_mpc_horizon_once.py
```

The first horizon MPC now uses a single-rigid-body approximation:

```text
state   = [com_pos, com_vel, theta, omega]
control = [f_FL, f_FR, f_RL, f_RR] at each horizon step

p[k+1] = p[k] + dt v[k]
v[k+1] = v[k] + dt (sum_i f_i[k] / m + g)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I^-1 sum_i (p_i - com) x f_i[k]
```

Here `theta` is a small-angle orientation error around the current MPC
linearization pose. The robot state still comes from MuJoCo's floating-base
quaternion; the QP does not optimize a quaternion directly.

with contact-schedule constraints:

```text
stance foot: friction pyramid and fz >= 0
swing foot: f = 0
```

Verify the horizon MPC first force sample as a WBC force reference:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_horizon_mpc_to_wbc_once.py
```

Inspect the per-knot contact schedule used by SRB-MPC:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_contact_schedule.py
```

Inspect the first quasi-static crawl task planner:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_crawl_planner.py
```

The SRB-MPC forward-step examples build `contact_schedule[k, foot]` from future
knot times:

```text
t_k = current_time + k * mpc_dt

stance foot: friction pyramid, fz >= normal_force_min
swing foot:  fx = fy = fz = 0
```

Run the SRB-MPC force reference in the forward-step WBC loop:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_srb_mpc_forward_step.py
```

Open the same SRB-MPC + WBC loop in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_srb_mpc_forward_step_viewer.py
```

Run the experimental slow crawl contact-sequence test:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_srb_mpc_crawl.py
```

Open the same crawl planner + SRB-MPC + WBC loop in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_srb_mpc_crawl_viewer.py
```

This currently exercises the `FL -> RR -> FR -> RL` contact schedule and WBC
task switching. The crawl-in-place version is a planning-layer smoke test, not
yet a forward walking gait.

Run one commanded forward crawl cycle:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_srb_mpc_crawl_forward.py
```

Open the commanded forward crawl in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_srb_mpc_crawl_forward_viewer.py
```

Run a short multi-cycle continuous forward crawl:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_srb_mpc_crawl_continuous.py
```

Open the continuous forward crawl in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_srb_mpc_crawl_continuous_viewer.py
```

The continuous crawl uses the first split planning stack:

```text
CrawlGaitPlanner        -> swing windows and contact schedule
RollingFootholdPlanner  -> rolling foot targets and locked stance positions
BodyReferencePlanner    -> conservative support-centroid body reference
ReferenceBundle         -> explicit base refs for WBC and COM refs for MPC
```

Solve one three-stance plus FL-swing WBC QP:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_single_leg_swing_once.py
```

Run a headless FL swing-in-place smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_single_leg_swing.py
```

Run the FL swing-in-place WBC in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_single_leg_swing_viewer.py
```

Run a headless FL forward-step foothold test:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_single_leg_forward_step.py
```

Run the FL forward-step WBC in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_single_leg_forward_step_viewer.py
```

The forward-step example uses three phases:

```text
1. four-foot stance with a small body shift
2. FR/RL/RR stance + FL swing to a forward foothold
3. FL touchdown, then FL/FR/RL/RR stance again
```

Run an experimental slow four-leg step sequence:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_step_sequence.py
```

Run the experimental step sequence in the MuJoCo viewer:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_step_sequence_viewer.py
```

The step-sequence scripts are intentionally experimental. The single-leg
step primitive is stable, but a full four-leg sequence still needs a proper
body reference scheduler before it should be treated as a gait.
