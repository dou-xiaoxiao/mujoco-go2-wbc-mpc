# Mainline Architecture

The main data flow is:

```text
MuJoCo Go2 state
    -> reference and contact schedule
    -> SRB-MPC
    -> full-body WBC QP
    -> joint torque
    -> MuJoCo step
```

## Data Flow

```text
MuJoCoModelInterface
    reads qpos, qvel, and contacts
    computes M, h, B, J, Jdot*v, COM, and inertia

Reference layer
    provides COM and base references
    provides stance/swing contact schedules
    provides swing-foot trajectories

CentroidalMPC
    optimizes horizon COM state and per-foot contact forces
    outputs the current per-foot force reference

GeneralContactWBCQP
    enforces full-body floating-base dynamics
    enforces stance acceleration constraints
    tracks swing-foot acceleration tasks
    tracks the MPC force reference as a soft cost
    outputs joint torque tau
```

## Main Demo Path

The main demo script is:

```text
scripts/record_trot_demo.py --preset trot-l-turn-stop
```

The script runs the controller headlessly and stores `qpos/qvel`, then replays
or renders the stored motion. This separates controller computation speed from
visual playback smoothness.

Current route:

```text
straight walking
left turn
short recovery pause
additional straight walking
final stop
```

Current gait mode:

```text
diagonal trot contact schedule
```

Current limitation:

```text
The route is scripted and conservative. It demonstrates the MPC/WBC stack, not
a production gait planner.
```

## Why MPC and WBC Are Both Present

The MPC sees a simplified centroidal model:

```text
state   = [com_pos, com_vel, theta, omega]
control = per-foot contact forces
```

It chooses contact forces over a prediction horizon while respecting a
time-varying contact schedule.

The WBC sees the full floating-base dynamics:

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
```

It maps base, swing-foot, and contact-force references into physically feasible
joint torques under stance, friction, and actuator constraints.

The split is:

```text
MPC: plan centroidal motion and contact-force references
WBC: realize those references using the full robot dynamics
```

## Current Research Boundary

This repository is currently an optimization-control locomotion prototype.

In scope:

```text
MuJoCo model interface
SRB-MPC
full-body WBC
contact schedules
scripted stance/swing references
stable replay demos
WBC timing instrumentation
```

Out of scope for the current repository:

```text
large-scale RL
Isaac Lab training
hardware deployment
terrain perception
fast dynamic gait optimization
```

These can be future extensions after the MPC/WBC stack is moved into a more
real-time-oriented C++ implementation.
