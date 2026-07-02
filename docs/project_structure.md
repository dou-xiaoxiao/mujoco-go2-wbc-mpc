# Project Structure / 读码顺序

这个项目按“核心库”和“运行入口”分开：

```text
src/mujoco_wbc/  可复用控制库
scripts/         可执行脚本和 demo
models/          MuJoCo XML / mesh 资源
docs/            架构、数学接口、读码说明
```

## 1. 核心库

```text
src/mujoco_wbc/model_interface.py
```

MuJoCo 包装层。负责从仿真中拿到控制器需要的动力学和运动学量：

```text
q, v
M(q)
h(q,v)
B
J_foot
Jdot_foot * v
COM
base rotation
foot contact positions
```

这是 MuJoCo 和控制数学之间的接口。

```text
src/mujoco_wbc/centroidal_mpc.py
```

SRB-MPC。用简化的 centroidal / single rigid body 模型优化预测窗口内的
COM 状态和每条腿接触力。

```text
src/mujoco_wbc/wbc_qp.py
```

Full-body WBC QP。决策变量是：

```text
z = [vdot, tau, f]
```

它在完整 floating-base 动力学约束下求关节力矩 `tau`。

```text
src/mujoco_wbc/contact_schedule.py
```

接触相位工具。负责表达哪些脚是 stance，哪些脚是 swing。

```text
src/mujoco_wbc/swing_trajectory.py
```

足端 swing 轨迹。输入起点、落点、抬脚高度、时间，输出足端位置、速度、加速度参考。

```text
src/mujoco_wbc/planning.py
```

简单 crawl planner。它不是成熟 gait planner，而是提供一条从速度命令到 foothold /
body reference 的最小链路。

```text
src/mujoco_wbc/reference_inputs.py
src/mujoco_wbc/support_polygon.py
src/mujoco_wbc/profiling.py
src/mujoco_wbc/conventions.py
```

辅助模块：reference 数据结构、支撑几何、计时 profiler、坐标约定。

## 2. 正式脚本入口

环境和模型检查：

```text
scripts/check_mujoco_install.py
scripts/launch_go2_viewer.py
scripts/inspect_go2_dynamics.py
scripts/inspect_frame_conventions.py
```

控制器回归：

```text
scripts/validate_control_stack.py
```

基础 WBC demo：

```text
scripts/run_static_stance_once.py
scripts/run_static_stance_viewer.py
scripts/run_single_leg_swing_once.py
scripts/run_single_leg_swing_viewer.py
```

主要 locomotion demo：

```text
scripts/record_trot_demo.py
scripts/run_trot_reference_viewer.py
scripts/run_commanded_crawl_viewer.py
```

`record_trot_demo.py` 是当前最适合展示的入口。它先离线 rollout，再固定帧率回放，
避免 live viewer 被 Python QP 求解拖慢。

## 3. 推荐阅读顺序

1. `README.md`
2. `docs/python_code_reading_guide.md`
3. `docs/control_stack.md`
4. `scripts/validate_control_stack.py`
5. `src/mujoco_wbc/model_interface.py`
6. `src/mujoco_wbc/wbc_qp.py`
7. `src/mujoco_wbc/centroidal_mpc.py`
8. `src/mujoco_wbc/swing_trajectory.py`
9. `scripts/run_trot_reference_viewer.py`
10. `scripts/record_trot_demo.py`

读代码时始终按这几个问题看：

```text
state 是什么？
input 是什么？
decision variable 是什么？
cost 是什么？
constraint 是什么？
output 给谁？
```

对应到本项目：

```text
model_interface.py
  input: MuJoCo model/data
  output: dynamics and kinematics quantities

centroidal_mpc.py
  input: COM reference, contact schedule, current robot state
  decision: horizon states and foot forces
  output: first-knot per-foot force reference

wbc_qp.py
  input: full dynamics, contact mode, swing/base references, MPC force reference
  decision: [vdot, tau, f]
  output: joint torque tau

viewer/demo scripts
  input: command/reference parameters
  output: MuJoCo simulation or replay
```

## 4. Python 结构速记

字典常用于每条腿的数据：

```python
foot_positions = {
    foot: robot.geom_position(foot)
    for foot in FOOT_GEOMS
}
```

元组常用于固定顺序：

```python
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
```

dataclass 常用于配置、窗口、求解结果：

```python
@dataclass(frozen=True)
class DemoWindow:
    swing_feet: tuple[str, ...]
    start_time: float
    duration: float
```

NumPy 切片常用于 floating-base 状态：

```python
base_pos = qpos[0:3]
base_quat = qpos[3:7]
joint_pos = qpos[7:]
```

`if __name__ == "__main__": main()` 表示：只有直接运行这个 `.py` 文件时才执行
`main()`；被其他文件 import 时不会自动运行。
