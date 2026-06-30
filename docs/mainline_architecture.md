# Mainline Architecture

项目主链路：

```text
MuJoCo Go2 state
    -> reference / contact schedule
    -> SRB-MPC
    -> full-body WBC QP
    -> joint torque
    -> MuJoCo step
```

## Data Flow

```text
MuJoCoModelInterface
    reads qpos, qvel, contacts
    computes M, h, B, J, Jdot*v, COM, inertia

Reference layer
    provides COM/base references
    provides stance/swing schedule
    provides swing foot trajectories

CentroidalMPC
    optimizes horizon COM state and foot forces
    outputs current per-foot force reference

GeneralContactWBCQP
    enforces full-body dynamics
    enforces stance acceleration constraints
    tracks swing foot acceleration tasks
    tracks MPC force reference as a soft cost
    outputs joint torque tau
```

## Main Demo Path

The clean public demo is:

```text
scripts/record_trot_demo.py --preset trot-l-route
```

It runs the controller headlessly and stores `qpos/qvel`, then replays or renders
the stored motion. This separates controller computation speed from visual playback
smoothness.

Current route:

```text
straight
left turn
short recovery pauses
straight
```

Current gait mode:

```text
diagonal trot contact schedule
```

Current limitation:

```text
The route is scripted and conservative. It demonstrates the MPC/WBC stack, not a
production gait planner.
```

## Why MPC and WBC Are Both Present

MPC sees a simplified centroidal model:

```text
state   = [com_pos, com_vel, theta, omega]
control = per-foot contact forces
```

It is good at choosing contact forces over a prediction horizon.

WBC sees the full floating-base dynamics:

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
```

It is good at turning references and contact forces into physically feasible
joint torques under stance, friction, and actuator constraints.

So the split is:

```text
MPC: plan centroidal motion and contact force reference
WBC: realize that reference using the full robot dynamics
```

## Current Research Boundary

This repository is currently an optimization-control locomotion project.

In scope:

```text
MuJoCo model interface
SRB-MPC
full-body WBC
contact schedules
scripted stance/swing references
stable replay demos
```

Out of scope for the current stable branch:

```text
large-scale RL
Isaac Lab training
hardware deployment
terrain perception
fast dynamic gait optimization
```

These can be future extensions once the current MPC/WBC project is packaged and
explained clearly.
