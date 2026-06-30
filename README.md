# MuJoCo Go2 SRB-MPC + Full-Body WBC Locomotion

这是一个从零搭建的四足机器人运动控制项目，目标是用 MuJoCo 的
floating-base 动力学接口实现 Unitree Go2 的 SRB-MPC + full-body WBC
控制栈。

项目当前不是为了追求最快、最炫的步态，而是为了把四足运动控制里最核心的
数学接口跑通并整理清楚：

```text
MuJoCo floating-base dynamics
    -> centroidal / single-rigid-body MPC
    -> full-body WBC QP
    -> joint torque command
```

![Trot L-route demo](docs/assets/trot_l_route_demo.gif)

## Highlights

- 使用 MuJoCo Go2 floating-base 模型，`nq=19, nv=18, nu=12`
- 明确 `qpos/qvel`、世界坐标系、base 坐标系、足端 Jacobian 约定
- 从 MuJoCo 读取 `M(q), h(q,v), B, J, Jdot*v`
- 实现 SRB-MPC，优化每条腿的世界系接触力参考
- 实现 full-body WBC QP，决策变量为 `[vdot, tau, f]`
- 支持四脚 stance、单腿 swing、任意非腾空接触模式的 generic WBC
- 支持 crawl 和 diagonal trot 的 reference tracking
- 提供稳定展示 demo：直走、左转 90 度、继续直走一段后停稳

## Current Demo

推荐给老师/面试官看的演示：

```powershell
cd D:\projects\quadruped_project\mujoco_wbc_project
.\.venv\Scripts\python.exe .\scripts\record_trot_demo.py --preset trot-l-turn-stop --no-gif --viewer-replay
```

这个 demo 会先离线 rollout 控制器，再用 MuJoCo viewer 平滑回放保存的状态。
这样即使 Python 版 MPC/WBC 求解速度低于实时，画面仍然是固定帧率回放。

当前稳定展示参数：

```text
preset       = trot-l-turn-stop
motion       = 直走 + 左转 90 度 + 继续直走一段 + 停稳
swing_height = 0.035 m
stance_gap   = 0.45 s
recovery     = 转弯后先恢复，再走一段，在下一次停顿处结束
foothold     = body-frame lateral width regulation
```

只看直走演示：

```powershell
.\.venv\Scripts\python.exe .\scripts\record_trot_demo.py --preset straight --no-gif --viewer-replay
```

生成 GIF：

```powershell
.\.venv\Scripts\python.exe .\scripts\record_trot_demo.py --preset trot-l-turn-stop
```

实时 viewer 版本，适合调试但可能因为 QP 求解显得卡：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_trot_reference_viewer.py --vx 0.012
```

## Installation

建议使用 Python 3.12 和虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Go2 模型来自 `mujoco_menagerie` 子模块。克隆仓库后需要初始化子模块：

```powershell
git submodule update --init --recursive
```

检查 MuJoCo 安装：

```powershell
.\.venv\Scripts\python.exe .\scripts\check_mujoco_install.py
```

打开 Go2 原始模型：

```powershell
.\.venv\Scripts\python.exe .\scripts\launch_go2_viewer.py
```

## Validation

快速回归测试：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\validate_control_stack.py
```

当前应通过：

```text
contact phase semantics
static stance WBC
single-leg swing WBC
general WBC crawl mode
general WBC trot mode
SRB-MPC all stance
SRB-MPC swing-foot force-zero
```

低帧率全路线稳定性检查：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\record_trot_demo.py --preset trot-l-route --no-gif --fps 1 --log-dt 20 --stop-on-fall
```

## Project Layout

```text
mujoco_wbc_project/
├── README.md
├── requirements.txt
├── models/
│   ├── free_body_smoke.xml
│   └── mujoco_menagerie/        # Git submodule, contains Unitree Go2 model
├── src/mujoco_wbc/
│   ├── model_interface.py       # MuJoCo dynamics / kinematics wrapper
│   ├── centroidal_mpc.py        # SRB-MPC contact-force QP
│   ├── wbc_qp.py                # full-body WBC QP
│   ├── contact_schedule.py      # stance/swing schedule helpers
│   ├── swing_trajectory.py      # swing foot trajectory
│   ├── planning.py              # simple crawl planner/reference bundle
│   ├── reference_inputs.py      # reference/contact-mode data structures
│   ├── support_polygon.py       # support geometry helpers
│   ├── profiling.py             # loop timing helper
│   └── conventions.py           # coordinate conventions
├── scripts/
│   ├── validate_control_stack.py
│   ├── record_trot_demo.py
│   ├── run_trot_reference_viewer.py
│   ├── run_commanded_crawl_viewer.py
│   ├── run_srb_mpc_crawl_continuous_viewer.py
│   ├── run_static_stance_once.py
│   ├── run_static_stance_viewer.py
│   ├── run_single_leg_swing_once.py
│   ├── run_single_leg_swing_viewer.py
│   ├── inspect_go2_dynamics.py
│   ├── inspect_frame_conventions.py
│   ├── launch_go2_viewer.py
│   └── check_mujoco_install.py
└── docs/
    ├── control_stack.md
    ├── mainline_architecture.md
    ├── project_structure.md
    └── locomotion_reference_map.md
```

## Mathematical Interface

### Generalized Coordinates

MuJoCo Go2 使用 floating-base 表示：

```text
qpos[0:3] = base position p_WB, expressed in world frame W
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity, expressed in W
qvel[3:6] = base angular velocity, expressed in W
qvel[6:]  = 12 joint velocities
```

因此：

```text
nq = 19
nv = 18
nu = 12
```

### WBC QP

WBC 的决策变量：

```text
z = [vdot, tau, f]
```

其中：

```text
vdot ∈ R^18   generalized acceleration
tau  ∈ R^12   joint torque
f    ∈ R^(3nc) stance foot contact forces, expressed in world frame
```

硬约束：

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
J_c(q) vdot + Jdot_c(q,v) v = a_c_cmd
tau_min <= tau <= tau_max
|fx| <= mu fz
|fy| <= mu fz
fz >= 0
```

软任务：

```text
base position / orientation tracking
nominal joint posture tracking
swing foot acceleration tracking
MPC contact force reference tracking
torque / force regularization
```

### SRB-MPC

MPC 使用 single rigid body / centroidal 近似：

```text
x = [com_pos, com_vel, theta, omega]
u = [f_FL, f_FR, f_RL, f_RR]
```

离散动力学：

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / m + g)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]
```

`theta` 是当前线性化附近的小角度姿态误差，不是全局四元数状态。

接触约束：

```text
stance foot: friction pyramid, fz >= normal_force_min
swing foot:  f = 0
```

MPC 输出当前 knot 的每足接触力参考，WBC 在完整 floating-base 动力学下尽量实现它。

## Current Scope and Limitations

已经完成并适合作为项目主体：

```text
MuJoCo Go2 model interface
SRB-MPC force planning
full-body WBC QP
generic contact-mode WBC
stance / single-leg swing / crawl / diagonal trot checks
stable route replay demo
```

当前不声称已经完成：

```text
real-time C++ implementation
hardware state estimator
sim2real
robust touchdown handling
fast natural trot
terrain locomotion
RL policy training
```

这些是后续方向，不是当前稳定展示主线。

## Possible Next Steps

适合毕业设计或后续研究的方向：

```text
1. Planner/reference layer: better foothold and body reference generation
2. Runtime: fixed sparsity QP update, C++ WBC/MPC implementation
3. Robustness: contact detection, load transfer, state estimation
4. RL hybrid: learn foothold residuals on top of MPC/WBC
5. Isaac Lab: large-scale locomotion RL training
```

当前建议先把本项目作为“基于动力学和优化控制的四足 locomotion 控制栈”展示。
