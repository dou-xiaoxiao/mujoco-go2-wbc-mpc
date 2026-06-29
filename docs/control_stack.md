# Control Stack Notes

This project currently separates the control stack into three layers:

```text
contact/gait schedule -> centroidal SRB-MPC -> full-body WBC QP -> MuJoCo torque control
```

The purpose of this document is to keep the stable control interfaces clear
while the higher-level gait logic is still experimental.

## Coordinate Conventions

The MuJoCo model is a floating-base Go2:

```text
qpos[0:3] = base position in world
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity in world
qvel[3:6] = base angular velocity in world
qvel[6:]  = 12 joint velocities
```

Foot Jacobians are expressed in the world frame:

```text
v_foot = J_foot(q) v
a_foot = J_foot(q) vdot + Jdot_foot(q, v) v
```

Contact forces are also world-frame 3D forces at the foot point:

```text
generalized contact force = J_foot(q)^T f_foot
```

## WBC QP Interface

The WBC decision variable is:

```text
z = [vdot, tau, f]
```

For stance feet, WBC enforces full rigid-body dynamics:

```text
M(q) vdot + h(q, v) = B tau + J_c(q)^T f
```

and stance foot acceleration constraints:

```text
J_c vdot + Jdot_c v = a_c_cmd
```

For the default static stance, `a_c_cmd = 0`. When `stance_pos_refs` is passed,
the stance constraint becomes a hard acceleration command:

```text
a_c_cmd = kp_stance (p_ref - p_foot) + kd_stance (0 - v_foot)
```

Swing feet are soft acceleration tasks:

```text
J_sw vdot + Jdot_sw v ~= xddot_ref
```

The WBC can also receive a contact-force reference from MPC. This is not a hard
constraint; it is a cost term. The hard constraint remains the full-body
dynamics equation above.

## SRB-MPC Interface

The horizon MPC uses a single-rigid-body centroidal model. Its state is:

```text
x = [com_pos, com_vel, theta, omega]
```

and its control is:

```text
u[k] = [f_FL, f_FR, f_RL, f_RR]
```

The dynamics are:

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / mass + gravity)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_world^-1 sum_i (p_i - com) x f_i[k]
```

Here `theta` is a local small-angle orientation error inside the QP. The actual
robot pose still comes from MuJoCo's quaternion state.

The MPC input `contact_schedule[k, foot]` has this meaning:

```text
True  = stance foot, friction pyramid is active
False = swing foot, force is constrained to zero
```

The first MPC force sample is passed to WBC as `force_ref`.

## Stable Baseline

The stable baseline at this point is:

```text
1. Static four-foot stance WBC
2. Three-foot stance plus FL swing WBC
3. FL forward-step WBC with touchdown and re-stance
4. SRB-MPC force reference connected into the FL forward-step WBC loop
```

Use this command as the quick regression check:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_control_stack.py
```

For a longer closed-loop check:

```powershell
.\.venv\Scripts\python.exe .\scripts\simulate_srb_mpc_forward_step.py
```

## Experimental Items

The crawl and four-leg step sequence scripts are intentionally experimental.
They exercise contact schedule switching, but they are not yet stable gaits.

Current known limitations:

```text
1. No proper body-reference scheduler above MPC yet.
2. No foothold planner beyond simple scripted swing targets.
3. Horizon MPC freezes current contact positions across the horizon.
4. Touchdown is currently based on foot height/time logic, not measured normal force.
5. MPC is rebuilt at each update instead of using OSQP matrix updates.
```

These limitations are above the basic MPC/WBC interface. They should be fixed
as gait-planning tasks rather than hidden by tuning the QP weights.
