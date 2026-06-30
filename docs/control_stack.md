# Control Stack / 控制栈说明

当前主线：

```text
planner / reference
    -> SRB-MPC
    -> full-body WBC QP
    -> MuJoCo torque control
```

这个文档只说明稳定主链路，不包含已经删除的早期实验脚本。

## 1. MuJoCo State

Unitree Go2 是 floating-base 模型：

```text
qpos[0:3] = base position, expressed in world frame
qpos[3:7] = base quaternion [w, x, y, z]
qpos[7:]  = 12 joint positions

qvel[0:3] = base linear velocity, expressed in world frame
qvel[3:6] = base angular velocity, expressed in world frame
qvel[6:]  = 12 joint velocities
```

维度：

```text
nq = 19
nv = 18
nu = 12
```

足端 Jacobian 约定：

```text
v_foot = J_foot(q) v
a_foot = J_foot(q) vdot + Jdot_foot(q,v) v
```

接触力约定：

```text
f_foot = [fx, fy, fz], expressed in world frame
generalized contact force = J_foot(q)^T f_foot
```

## 2. Planner / Reference Layer

这个项目的 planner 很薄，只负责给 MPC/WBC 提供 reference：

```text
contact schedule
swing foot start / target foothold
swing foot position / velocity / acceleration reference
base position / orientation reference
COM position / velocity reference
```

目前最稳定的 trot route demo 仍然是脚本化 reference，不是成熟自主 gait planner：

```text
直走
左转
短暂停顿恢复
继续直走
```

这也是项目当前的边界：控制层已经搭好，planner 仍然是后续工作。

## 3. SRB-MPC

文件：

```text
src/mujoco_wbc/centroidal_mpc.py
```

SRB = single rigid body。MPC 使用整机 COM 和简化姿态动力学：

```text
x = [com_pos, com_vel, theta, omega]
u = [f_FL, f_FR, f_RL, f_RR]
```

优化变量：

```text
X[0:N]    predicted centroidal states
F[0:N-1]  predicted contact forces
```

动力学：

```text
p[k+1]     = p[k] + dt v[k]
v[k+1]     = v[k] + dt (sum_i f_i[k] / mass + gravity)
theta[k+1] = theta[k] + dt omega[k]
omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]
```

约束：

```text
stance foot:
  |fx| <= mu fz
  |fy| <= mu fz
  fz >= normal_force_min

swing foot:
  fx = fy = fz = 0
```

代价函数：

```text
track COM position
track COM velocity
track small-angle orientation
track angular velocity
regularize contact force
regularize force rate over horizon
```

输出：

```text
first-knot per-foot force reference
```

MPC 的 `f_i` 不是直接施加到仿真里的力，而是传给 WBC 的接触力参考。

## 4. Full-Body WBC QP

文件：

```text
src/mujoco_wbc/wbc_qp.py
```

决策变量：

```text
z = [vdot, tau, f]
```

含义：

```text
vdot ∈ R^18    generalized acceleration
tau  ∈ R^12    joint torque
f    ∈ R^(3nc) stance foot contact forces
```

硬动力学约束：

```text
M(q) vdot + h(q,v) = B tau + J_c(q)^T f
```

stance 约束：

```text
J_c(q) vdot + Jdot_c(q,v) v = a_c_cmd
```

其中 `a_c_cmd` 可以是 0，也可以是足端位置反馈生成的加速度命令：

```text
a_c_cmd = kp (p_ref - p_foot) + kd (0 - v_foot)
```

swing 足是软任务：

```text
J_sw vdot + Jdot_sw v ~= xddot_ref
```

摩擦和力矩约束：

```text
|fx| <= mu fz
|fy| <= mu fz
fz >= 0
tau_min <= tau <= tau_max
```

代价函数：

```text
base position/orientation acceleration tracking
joint posture tracking
swing foot acceleration tracking
MPC force reference tracking
torque regularization
contact force regularization
```

输出：

```text
tau
```

## 5. Contact Modes

四脚 stance：

```text
MPC: all four feet can generate force
WBC: all four feet are stance constraints
```

单腿 swing：

```text
MPC: swing foot force is zero
WBC: swing foot leaves contact constraints and becomes a swing task
```

Diagonal trot：

```text
MPC: two stance feet can generate force, two swing feet are force-zero
WBC: two stance constraints + two swing tasks
```

当前 `GeneralContactWBCQP` 支持任意非腾空接触模式，所以 crawl 和 trot 都使用同一类
WBC，而不是为每种 gait 写一套独立控制器。

## 6. Stable Baseline

当前稳定基线：

```text
1. static four-foot stance WBC
2. single-leg swing WBC
3. general contact WBC: crawl mode
4. general contact WBC: diagonal trot mode
5. SRB-MPC force reference with time-varying contact schedule
6. offline trot route replay demo
```

回归命令：

```powershell
.\.venv\Scripts\python.exe -B .\scripts\validate_control_stack.py
```

展示命令：

```powershell
.\.venv\Scripts\python.exe .\scripts\record_trot_demo.py --preset trot-l-route --no-gif --viewer-replay
```

## 7. Known Limitations

当前没有声称解决：

```text
real-time performance
hardware state estimation
sim2real
robust touchdown/load transfer
natural high-speed trot
terrain adaptation
RL policy training
```

这些是后续方向。当前项目的价值是：完整、可解释地搭出 MPC/WBC locomotion 控制链路。
