# MPC/WBC Mainline Architecture

This file is the short map of the current project. It deliberately ignores
most tuning details and keeps only the control pipeline:

```text
MuJoCo robot state
  -> planner / references
  -> centroidal SRB-MPC
  -> full-body WBC QP
  -> joint torques
  -> MuJoCo simulation
```

## 1. Robot State and Dynamics

Owner:

```text
src/mujoco_wbc/model_interface.py
```

State:

```text
qpos[0:3] = base position in world
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity in world
qvel[3:6] = base angular velocity in world
qvel[6:]  = 12 joint velocities
```

It provides the data needed by WBC:

```text
M(q)
h(q, v)
B
J_foot(q)
Jdot_foot(q, v) v
base orientation R_WB
COM position / velocity
composite inertia
foot contact-point positions
```

The key convention is:

```text
v_foot = J_foot(q) v
a_foot = J_foot(q) vdot + Jdot_foot(q, v) v
generalized contact force = J_foot(q)^T f_foot
```

No IK is in the main locomotion chain. Foot tasks are expressed directly as
task-space acceleration tasks through the Jacobian.

## 2. Planner and References

Owner:

```text
src/mujoco_wbc/planning.py
src/mujoco_wbc/contact_schedule.py
src/mujoco_wbc/swing_trajectory.py
```

Human/task input:

```text
CrawlCommand(vx, vy, yaw_rate)
```

Planner outputs:

```text
contact schedule over the MPC horizon
which foot is swing at the current time
swing foot start position
swing foot target foothold
swing foot position / velocity / acceleration reference
base position / orientation reference for WBC
COM position / velocity reference for MPC
```

Important distinction:

```text
foothold target -> consumed by swing trajectory and WBC swing task
COM reference   -> consumed by MPC
base reference  -> consumed by WBC
```

The current planner is intentionally simple. It converts command velocity into
a small world-frame foothold delta, then biases the body reference toward the
upcoming support set. This is the upper layer that is still least mature.

## 3. Centroidal SRB-MPC

Owner:

```text
src/mujoco_wbc/centroidal_mpc.py
```

SRB means single rigid body. The MPC does not optimize the full 18-velocity
floating-base dynamics. It uses a centroidal approximation:

```text
state x = [com_pos, com_vel, theta, omega]
input u = [f_FL, f_FR, f_RL, f_RR]
```

Decision variable:

```text
all states over the horizon
all four foot-force vectors over the horizon
```

Dynamics constraints:

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / mass + gravity)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_world^-1 sum_i (p_i - com) x f_i[k]
```

Contact constraints:

```text
stance foot: friction pyramid and fz >= 0
swing foot:  f = 0
```

Cost:

```text
track COM position reference
track COM velocity reference
keep small orientation error
keep small angular velocity
regularize contact forces
```

Output to WBC:

```text
current-knot per-foot contact force reference
```

This force is a reference, not a torque command. MPC says what contact forces
would produce a good centroidal motion under the simplified model.

## 4. Full-Body WBC QP

Owner:

```text
src/mujoco_wbc/wbc_qp.py
```

Decision variable:

```text
z = [vdot, tau, f]
```

where:

```text
vdot = generalized acceleration, R^18
tau  = 12 joint torques
f    = stance foot contact forces
```

Hard dynamics constraint:

```text
M(q) vdot + h(q, v) = B tau + J_c(q)^T f
```

Hard stance-foot acceleration constraint:

```text
J_c vdot + Jdot_c v = a_c_cmd
```

For a normal locked stance foot:

```text
a_c_cmd = kp (p_ref - p_foot) + kd (0 - v_foot)
```

Swing-foot soft task:

```text
J_sw vdot + Jdot_sw v ~= xddot_ref
```

Force and actuator constraints:

```text
|fx| <= mu fz
|fy| <= mu fz
fz >= 0
tau_min <= tau <= tau_max
```

Cost:

```text
track base position/orientation acceleration task
track nominal joint posture
track swing foot acceleration task when a foot is swinging
track MPC contact-force reference
regularize torques
regularize contact forces during landing load transfer
```

Output:

```text
tau
```

WBC is the layer that makes the MPC force reference physically compatible with
the real floating-base dynamics, stance constraints, friction constraints, and
torque limits.

## 5. Contact Mode Changes

Four-foot stance:

```text
MPC: all four feet may generate force
WBC: all four feet are stance constraints
```

One-foot swing:

```text
MPC: swing foot force is constrained to zero over knots where it is swing
WBC: swing foot is removed from contact constraints and added as a swing task
```

After touchdown:

```text
planner locks the new foothold
MPC allows that foot to generate force again
WBC puts that foot back into stance
landing load transfer ramps the newly touched foot force in smoothly
```

The recent touchdown and load-transfer code is engineering around this mode
switch. It is not a new control layer.

## 6. What Is Done vs Not Done

Done enough for the current milestone:

```text
MuJoCo Go2 floating-base model interface
full-body WBC dynamics QP
single-leg swing WBC without IK
SRB-MPC force references
time-varying contact schedule over the MPC horizon
continuous crawl demo path in the viewer
QP solver reuse for MPC, stance WBC, and swing WBC
basic profiler for viewer / MPC / WBC timing
```

Still experimental:

```text
foothold quality
body / COM reference quality
touchdown smoothness
turning and lateral walking
faster real-time performance
trot or dynamic gaits
hardware estimator / IMU / state-estimation interface
```

The next useful project step is not more hidden tuning. It is to make the demo
entry point cleaner, expose command parameters clearly, and then improve the
planner/reference layer in small visible increments.
