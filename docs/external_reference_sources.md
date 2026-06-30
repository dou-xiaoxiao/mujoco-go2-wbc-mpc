# External Reference Sources

This project does not aim to build a smart planner. The intended boundary is:

```text
upstream planner / VLA / WAM / RL policy
  -> locomotion reference
  -> MPC/WBC torque-control stack
```

The reference that the current control stack wants is:

```text
contact schedule
foothold targets
swing foot references
base reference
COM reference
velocity command
```

## Sources Worth Connecting

### Kine2Go

Kine2Go is a recent Unitree Go2 motion dataset. It is the most directly useful
source for this project because it targets the same robot family and contains
kinematic trajectories plus motor-level action data.

Reference:

```text
https://arxiv.org/abs/2606.14433
```

Good use in this project:

```text
base / joint / foot reference extraction
contact inference from foot height or velocity
checking whether WBC can track externally provided motion clips
```

Expected gap:

```text
dataset clips must be converted into our LocomotionReferenceFrame format
multi-swing and flight clips require WBC modes we do not implement yet
```

### MIT Cheetah / Mini Cheetah Convex MPC Style Gaits

The useful part for us is not copying their planner, but using their gait
families as contact-mode tests:

Reference:

```text
https://github.com/mit-biomimetics/Cheetah-Software
```

```text
crawl
trot
pace
bound
pronk
```

Good use in this project:

```text
contact schedule validation
MPC horizon mode switching tests
deciding which WBC mode is missing next
```

Expected gap:

```text
dynamic quality for trot, pace, and bound depends on better references
pronk and jump require flight handling
```

### OCS2 Legged Robot

OCS2's legged robot stack is useful as a reference for mode schedules and the
separation between an optimizer/reference manager and a lower-level controller.

Reference:

```text
https://github.com/leggedrobotics/ocs2
```

Good use in this project:

```text
mode schedule interface design
contact switching tests over an MPC horizon
```

Expected gap:

```text
OCS2 is a larger NMPC stack; we only need its interface pattern, not its whole architecture
```

### Expressive Go2 Motion Datasets

Newer datasets such as language-annotated expressive Go2 motions are useful
later, but they are not the first target for this model-based control project.

Good use in this project:

```text
stress-testing reference ingestion
classifying which motions are feasible for the current WBC
```

Expected gap:

```text
many motions rely on learned tracking policies, retargeting, or dynamic phases
that our current WBC does not yet execute
```

## Current Executability

The current WBC supports:

```text
all four feet stance
one swing foot + three stance feet
two swing feet + two stance feet
generic non-flight stance/swing subsets
```

It does not yet support:

```text
zero stance feet / flight phase   -> needed for pronk / jump
```

Run:

```powershell
.\.venv\Scripts\python.exe .\scripts\inspect_external_reference_modes.py
```

This prints which contact modes can enter the current WBC and which modes need
new WBC support.

## Practical Next Step

Do not tune the planner further. The current control-layer feature is:

```text
GeneralContactWBCQP
```

This executor lets crawl-like and trot-like references enter the same MPC/WBC
pipeline without changing the project boundary. The remaining missing executor
for jumping is a flight-phase WBC, which is intentionally out of scope.
