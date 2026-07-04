# Control Stack

The stable control pipeline is:

```text
reference and contact schedule
    -> SRB-MPC
    -> full-body WBC QP
    -> MuJoCo torque control
```

This document describes the main optimized-control path used by the current
repository. It does not cover early experimental scripts or tuning notes.

## 1. MuJoCo State

The Unitree Go2 model is represented as a floating-base system:

```text
qpos[0:3] = base position, expressed in world frame
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity, expressed in world frame
qvel[3:6] = base angular velocity, expressed in world frame
qvel[6:]  = 12 joint velocities
```

Dimensions:

```text
nq = 19
nv = 18
nu = 12
```

Foot Jacobian convention:

```text
v_foot = J_foot(q) v
a_foot = J_foot(q) vdot + Jdot_foot(q,v) v
```

Contact-force convention:

```text
f_foot = [fx, fy, fz], expressed in world frame
generalized contact force = J_foot(q)^T f_foot
```

## 2. Reference Layer

The current reference layer is intentionally lightweight. It provides:

```text
contact schedule
swing-foot start and target foothold
swing-foot position, velocity, and acceleration reference
base position and orientation reference
COM position and velocity reference
```

The public route demo uses scripted references:

```text
straight walking
left turn
short recovery pause
additional straight walking
final stop
```

The planner is intentionally conservative and is not presented as a production
gait planner.

## 3. SRB-MPC

Implementation:

```text
src/mujoco_wbc/centroidal_mpc.py
```

The MPC uses a single-rigid-body model of the robot:

```text
x = [com_pos, com_vel, theta, omega]
u = [f_FL, f_FR, f_RL, f_RR]
```

Optimization variables:

```text
X[0:N]    predicted centroidal states
F[0:N-1]  predicted contact forces
```

Discrete dynamics:

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / mass + gravity)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]
```

Contact constraints:

```text
stance foot:
  |fx| <= mu fz
  |fy| <= mu fz
  fz >= normal_force_min

swing foot:
  fx = fy = fz = 0
```

Cost terms:

```text
COM position tracking
COM velocity tracking
small-angle orientation tracking
angular velocity tracking
contact-force regularization
force-rate regularization
```

Output:

```text
first-knot per-foot contact-force reference
```

The MPC force is not applied directly to the simulator. It is passed to the WBC
as a contact-force reference.

## 4. Full-Body WBC QP

Implementation:

```text
src/mujoco_wbc/wbc_qp.py
```

Decision variable:

```text
z = [vdot, tau, f]
```

where:

```text
vdot in R^18    generalized acceleration
tau  in R^12    joint torque
f    in R^(3nc) stance-foot contact forces
```

Floating-base dynamics constraint:

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
```

Stance acceleration constraint:

```text
J_c(q) vdot + Jdot_c(q,v) v = a_c_cmd
```

The stance acceleration command can be zero or a Cartesian foot-position
feedback command:

```text
a_c_cmd = kp (p_ref - p_foot) + kd (0 - v_foot)
```

Swing feet are tracked as soft acceleration tasks:

```text
J_sw vdot + Jdot_sw v ~= xddot_ref
```

Friction and torque constraints:

```text
|fx| <= mu fz
|fy| <= mu fz
fz >= 0
tau_min <= tau <= tau_max
```

Cost terms:

```text
base position and orientation acceleration tracking
joint posture tracking
swing-foot acceleration tracking
MPC force-reference tracking
torque regularization
contact-force regularization
```

Output:

```text
joint torque tau
```

## 5. Contact Modes

Four-foot stance:

```text
MPC: all four feet can generate contact force
WBC: all four feet are stance constraints
```

Single-leg swing:

```text
MPC: swing-foot force is constrained to zero
WBC: swing foot becomes a swing task and leaves the stance constraints
```

Diagonal trot:

```text
MPC: two stance feet can generate force, two swing feet are force-zero
WBC: two stance constraints and two swing tasks
```

`GeneralContactWBCQP` supports arbitrary non-flight contact modes, so crawl and
diagonal trot can share the same WBC implementation.

## 6. Stable Baseline

Current validated baseline:

```text
static four-foot stance WBC
single-leg swing WBC
general contact WBC in crawl mode
general contact WBC in diagonal trot mode
SRB-MPC force reference with time-varying contact schedules
offline trot route replay demo
```

Regression command:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\validate_control_stack.py
```

Replay command:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-turn-stop --no-gif --viewer-replay
```

## 7. Known Limitations

The current Python prototype does not claim to solve:

```text
real-time deployment
hardware state estimation
hardware sim-to-real transfer
robust touchdown and load transfer
natural high-speed trot
terrain adaptation
RL policy training
```

The value of the current repository is a compact, inspectable MPC/WBC
locomotion stack built directly on MuJoCo dynamics and kinematics.
