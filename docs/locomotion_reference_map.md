# Locomotion Reference Map

This project keeps the control stack split into:

```text
planner/reference layer -> centroidal MPC -> full-body WBC -> MuJoCo torque control
```

External projects are useful as interface references, not as code to copy blindly.

## External Patterns

MIT Cheetah / Mini Cheetah style convex MPC:

- The upper layer provides gait/contact timing and body motion targets.
- The MPC optimizes stance foot contact forces over a horizon.
- The whole-body or leg controller realizes those forces and swing trajectories.

OCS2 legged robot style NMPC:

- The optimizer can carry a richer centroidal state and mode schedule.
- Contact mode changes are explicit per horizon knot.
- The controller still needs a lower-level whole-body layer to map plans to torques.

Unitree / Go1 / Go2 open controllers:

- Practical deployments separate command, gait scheduling, swing trajectories, stance control, and estimator interfaces.
- The hardware-facing layer usually consumes joint torques or joint PD targets plus feedforward terms.

legged_gym style RL command interface:

- Commands are compact: forward velocity, lateral velocity, yaw rate, sometimes body height.
- Gait and footholds may be implicit in a learned policy, but the command interface is still a useful reference for hand-written planning.

## Interface We Will Use

`CrawlCommand` is the human/task command:

```text
vx
vy
yaw_rate
```

`CrawlGaitConfig` is the contact-mode schedule:

```text
sequence
first_swing_start
swing_duration
swing_gap
swing_height
step_delta
```

`RollingFootholdPlanner` owns foot targets:

```text
locked stance foot positions in world
active swing start position in world
active swing target position in world
swing reference position/velocity/acceleration
```

`BodyReferencePlanner` owns conservative body references:

```text
base_position_ref
base_orientation_ref
com_position_ref
com_velocity_ref
```

The current body reference is still deliberately quasi-static in position:

```text
body xy ref -> support-centroid-biased target for the upcoming stance triangle
```

When `CrawlCommand` is present, the planner also passes a small COM velocity
reference to MPC:

```text
com_velocity_ref = command_velocity_ref_scale * [vx, vy, 0]
```

`CentroidalMPC` consumes:

```text
current full robot state through MuJoCoModelInterface
com_position_ref
com_velocity_ref
contact_schedule over the horizon
```

and produces:

```text
per-foot contact force references for the current knot
```

`StanceWBCQP` / `SingleLegSwingWBCQP` consume:

```text
qpos_ref for base/joints
stance foot position references
swing foot task reference when one foot is in swing
MPC force reference for stance feet
```

and produce:

```text
joint torque command tau
```

## Command-Based Foothold Hook

The planner can keep the fixed `STEP_DELTA` rule, or use `CrawlCommand` to
generate command-based foothold deltas:

```text
target_foothold_W =
    nominal_foot_W
  + velocity_feedforward_W
  + yaw_feedforward_W
  + stability_bias_W
```

Current first version:

```text
velocity_feedforward_W = gait_cycle_duration * [vx, vy, 0]
yaw_feedforward_W      = yaw_rate * gait_cycle_duration * z_axis_cross_foot_offset
stability_bias_W       = small shift toward the support centroid
```

The current implementation includes velocity and yaw feedforward. The gait
cycle duration is based on one pass through the foot set, not the total number
of repeated cycles in a script.

The stability bias is intentionally left as a planner-tuning item, because it
should be tested against the MPC/WBC response rather than baked into the first
command interface.

This keeps the learning target clear:

```text
commands shape footholds and body refs
MPC shapes contact forces
WBC realizes forces, stance constraints, and swing tracking
```
