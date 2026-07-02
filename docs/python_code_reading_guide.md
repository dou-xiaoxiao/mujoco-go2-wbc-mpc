# Python Code Reading Guide

这份文档的目标不是教 Python 语法大全，而是帮你读懂当前项目代码。

你已经懂控制理论，所以读代码时应该一直问这几个问题：

```text
状态是什么？
输入是什么？
decision variable 是什么？
cost 是什么？
constraint 是什么？
输出给谁？
```

Python 只是把这些东西组织起来的工具。

## 1. 项目代码地图

当前主线分成三层：

```text
README.md / docs/
    给人看的解释

src/mujoco_wbc/
    真正的控制库

scripts/
    实验入口：打开 viewer、跑验证、录 GIF
```

最重要的代码顺序：

```text
src/mujoco_wbc/conventions.py
src/mujoco_wbc/model_interface.py
src/mujoco_wbc/wbc_qp.py
src/mujoco_wbc/centroidal_mpc.py
src/mujoco_wbc/contact_schedule.py
src/mujoco_wbc/swing_trajectory.py
src/mujoco_wbc/planning.py
scripts/validate_control_stack.py
scripts/run_trot_reference_viewer.py
scripts/record_trot_demo.py
```

先读：

```text
model_interface.py
wbc_qp.py
centroidal_mpc.py
run_trot_reference_viewer.py
```

这四个读通，项目主链路就通了。

## 2. import 到底是什么

例如 `scripts/run_trot_reference_viewer.py` 里有：

```python
from mujoco_wbc import (
    CentroidalMPC,
    CentroidalMPCConfig,
    GeneralContactWBCConfig,
    GeneralContactWBCQP,
    MuJoCoModelInterface,
    StanceWBCConfig,
    StanceWBCQP,
    swing_foothold_reference,
)
```

这不是把代码复制过来，而是：

```text
加载 mujoco_wbc 这个包
从里面拿出这些类/函数名字
让当前脚本可以直接使用它们
```

项目里的 `src/mujoco_wbc/__init__.py` 类似“对外导出表”。脚本不需要知道每个类具体在哪个文件，只要从 `mujoco_wbc` 导入即可。

例如：

```python
robot = MuJoCoModelInterface(MODEL_PATH)
mpc = CentroidalMPC(mpc_config)
wbc = GeneralContactWBCQP(wbc_config)
```

读法是：

```text
创建一个 MuJoCo 机器人接口对象
创建一个 MPC 控制器对象
创建一个 WBC 控制器对象
```

## 3. 路径和 sys.path

脚本里常见：

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "mujoco_menagerie" / "unitree_go2" / "scene.xml"

sys.path.insert(0, str(SRC_ROOT))
```

意思是：

```text
__file__              当前脚本文件路径
Path(...).resolve()   转成绝对路径
parents[1]            往上两级，得到项目根目录
PROJECT_ROOT / "src"  拼路径
sys.path.insert       告诉 Python 去 src 里找 mujoco_wbc 包
```

`/` 在 `Path` 对象里不是除法，而是拼路径。

## 4. np.ndarray 是项目里的主要数学对象

项目里：

```python
Array = np.ndarray
```

这只是类型别名。看到：

```python
def mass_matrix(self) -> Array:
```

就等价于：

```text
这个函数返回一个 numpy 数组
```

常见数组含义：

```text
qpos        MuJoCo generalized coordinate, shape = (19,)
qvel        MuJoCo generalized velocity, shape = (18,)
M           mass matrix, shape = (18, 18)
h           bias force, shape = (18,)
B           actuation matrix, shape = (18, 12)
J           foot Jacobian, shape = (3*num_feet, 18)
tau         joint torque, shape = (12,)
f           contact force, shape = (3*num_contacts,)
```

### 4.1 copy 很重要

项目里经常写：

```python
home_qpos_ref = robot.q.copy()
initial_base_pos = robot.data.qpos[0:3].copy()
```

原因是 numpy 数组经常是“视图”。如果你不 `.copy()`，后面 MuJoCo 状态变了，你保存的引用可能也跟着变，或者你改引用时把原数据也改了。

经验规则：

```text
只是读取一瞬间状态并长期保存：用 .copy()
临时计算：可以不用 .copy()
要写回 MuJoCo data：用 [:] 原地写入
```

例如：

```python
robot.data.ctrl[:] = last_tau
```

这里 `[:]` 的意思是：

```text
不要换掉 robot.data.ctrl 这个数组对象
而是把 last_tau 的数值写进已有数组
```

MuJoCo 的 `data.qpos/data.qvel/data.ctrl` 这种数组通常应该这样写。

## 5. tuple / list / set / dict

### 5.1 tuple：固定顺序，不打算改

项目里脚顺序经常是 tuple：

```python
FOOT_GEOMS = ("FL", "FR", "RL", "RR")
TROT_PAIRS = (("FL", "RR"), ("FR", "RL"))
```

读法：

```text
FOOT_GEOMS 固定四条腿顺序
TROT_PAIRS 固定 trot 对角腿顺序
```

tuple 用圆括号，通常代表：

```text
这组东西数量固定、顺序固定、不打算修改
```

### 5.2 list：可变序列

例如：

```python
windows: list[TrotWindow] = []
windows.append(TrotWindow(...))
```

list 用方括号，适合：

```text
时间窗口序列
采样数据
MPC horizon 上的多步数据
```

常见操作：

```python
windows.append(x)       # 末尾添加
windows[idx]            # 按下标取
len(windows)            # 长度
for window in windows:  # 遍历
```

### 5.3 set：只关心有没有，不关心顺序

例如：

```python
completed_windows: set[int] = set()
completed_windows.add(active_window_id)
```

set 是集合，适合：

```text
记录某个窗口是否完成
判断某个元素是否存在
```

常见操作：

```python
x in completed_windows      # 判断是否存在
completed_windows.add(x)    # 加进去
```

### 5.4 dict：本项目最常见的数据结构

dict 是键值表：

```python
locked_positions = {
    "FL": np.array([...]),
    "FR": np.array([...]),
    "RL": np.array([...]),
    "RR": np.array([...]),
}
```

含义：

```text
用 foot name 找对应数据
```

也就是：

```text
foot -> position
foot -> swing reference
foot -> controller
foot -> force
```

## 6. dict 语法细讲

### 6.1 创建字典

普通写法：

```python
d = {}
d["FL"] = np.array([0.2, 0.1, 0.0])
d["FR"] = np.array([0.2, -0.1, 0.0])
```

字面量写法：

```python
d = {
    "FL": np.array([0.2, 0.1, 0.0]),
    "FR": np.array([0.2, -0.1, 0.0]),
}
```

项目里常见的是 dict comprehension：

```python
initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
```

展开理解就是：

```python
initial_foot_positions = {}
for foot in FOOT_GEOMS:
    initial_foot_positions[foot] = robot.geom_position(foot)
```

所以这句的物理意义是：

```text
对 FL/FR/RL/RR 每条腿，读取当前足端位置，存成 foot -> position
```

### 6.2 读取一个键

```python
pos = locked_positions["FL"]
```

意思：

```text
从 locked_positions 这个字典里取 "FL" 对应的位置
```

如果 `"FL"` 不存在，会报错 `KeyError`。

### 6.3 安全读取 get

```python
value = d.get("FL", default_value)
```

意思：

```text
如果 "FL" 存在，就返回 d["FL"]
如果不存在，就返回 default_value
```

这个项目主链路里更常用明确的 `d[key]`，因为四条腿键应该存在；`get` 更适合可选参数和调试代码。

### 6.4 修改一个键对应的值

```python
locked_positions[foot] = active_plans[foot].target_position.copy()
```

意思：

```text
把 foot 这条腿的 locked stance position 更新为 swing 目标落点
```

这是“换掉这个 key 对应的 value”。

### 6.5 修改 value 里面的数组

如果 value 是 numpy 数组：

```python
locked_positions["FL"][2] += 0.01
```

这不是换掉 value，而是改 value 数组里的第 3 个元素。

区别：

```python
locked_positions["FL"] = new_array
```

是让 `"FL"` 指向一个新数组。

```python
locked_positions["FL"][2] = 0.0
```

是修改原数组的 z 分量。

因为 numpy 数组是可变对象，所以项目里经常用 `.copy()` 避免不小心共享同一个数组。

例如：

```python
locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}
```

这里如果不 copy，`locked_positions["FL"]` 和 `initial_foot_positions["FL"]` 可能指向同一个数组，后面改一个会影响另一个。

### 6.6 判断 key 是否存在

```python
if key not in generic_controllers:
    generic_controllers[key] = GeneralContactWBCQP(...)
```

意思：

```text
如果这个接触模式对应的 WBC 控制器还没创建，就创建一个并缓存
```

这里的 key 是：

```python
key = (stance_feet, swing_feet)
```

类型是：

```text
tuple[tuple[str, ...], tuple[str, ...]]
```

例如：

```python
key = (("FR", "RL"), ("FL", "RR"))
```

物理意义：

```text
stance feet = FR, RL
swing feet  = FL, RR
```

缓存的意义：

```text
同一种 contact mode 复用同一个 WBCQP 对象
```

### 6.7 遍历 keys

```python
for foot in FOOT_GEOMS:
    pos = locked_positions[foot]
```

这里不是遍历字典本身，而是按固定脚顺序访问字典。这样顺序可控。

### 6.8 遍历 items

```python
for foot, pos in initial_foot_positions.items():
    locked_positions[foot] = pos.copy()
```

`.items()` 每次给你一对：

```text
key, value
```

也就是：

```text
foot, position
```

项目例子：

```python
swing_refs = {
    foot: swing_foothold_reference(...)
    for foot, plan in active_plans.items()
}
```

意思：

```text
遍历 active_plans 里的每条 swing 腿
对每条腿生成 swing reference
得到 foot -> SwingReference
```

### 6.9 遍历 values

```python
center_xy = np.mean(np.vstack([pos[0:2] for pos in initial_foot_positions.values()]), axis=0)
```

`.values()` 只取 value，不取 key。

这里意思是：

```text
拿到四个初始足端位置
取每个位置的 x,y
堆成矩阵
求平均，得到足端几何中心
```

### 6.10 dict comprehension

常见形式：

```python
{key_expr: value_expr for item in iterable}
```

项目里：

```python
swing_pos_refs={foot: ref.position for foot, ref in swing_refs.items()}
```

展开：

```python
swing_pos_refs = {}
for foot, ref in swing_refs.items():
    swing_pos_refs[foot] = ref.position
```

物理意义：

```text
WBC 不需要完整 SwingReference 对象
它只需要 foot -> desired position
```

类似还有：

```python
swing_vel_refs={foot: ref.velocity for foot, ref in swing_refs.items()}
swing_acc_refs={foot: ref.acceleration for foot, ref in swing_refs.items()}
stance_pos_refs={foot: locked_positions[foot] for foot in stance_feet}
```

这些都是把“总信息”拆成 WBC solve 需要的输入。

## 7. dataclass 是什么

项目里大量使用：

```python
@dataclass(frozen=True)
class CentroidalMPCConfig:
    contact_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    horizon_steps: int = 10
    dt: float = 0.03
```

dataclass 的作用：

```text
自动帮你写 __init__
让你可以用对象名.字段名访问数据
适合 config / result / reference
```

例如：

```python
cfg = CentroidalMPCConfig(horizon_steps=12, dt=0.03)
```

然后：

```python
cfg.horizon_steps
cfg.dt
cfg.contact_geoms
```

`frozen=True` 的意思是：

```text
创建后不应该再改字段
```

例如：

```python
cfg.dt = 0.05
```

会报错。

这很适合 config，因为控制器参数不应该在函数内部乱改。

### 7.1 Config / Solution / Reference

项目中 dataclass 大致分三类。

Config：

```text
CentroidalMPCConfig
StanceWBCConfig
GeneralContactWBCConfig
```

表示控制器参数，例如权重、摩擦系数、horizon。

Solution：

```text
CentroidalMPCSolution
StanceWBCSolution
GeneralContactWBCSolution
```

表示 solver 输出，例如状态、接触力、tau、residual。

Reference / Plan：

```text
SwingReference
SwingPlan
TrotWindow
ReferenceBundle
```

表示 planner 产生的目标轨迹或时间窗口。

读代码时看到 dataclass，你先问：

```text
这是输入参数？
这是输出结果？
这是中间计划？
```

## 8. class / self 是什么

例如：

```python
class MuJoCoModelInterface:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(...)
        self.data = mujoco.MjData(self.model)
```

读法：

```text
MuJoCoModelInterface 是一个对象类型
__init__ 是构造函数
self 代表当前这个对象
self.model / self.data 是对象长期持有的状态
```

当你写：

```python
robot = MuJoCoModelInterface(MODEL_PATH)
```

会创建一个对象，里面持有：

```text
robot.model
robot.data
robot.base_body_id
```

之后：

```python
robot.mass_matrix()
robot.bias_forces()
robot.geom_position("FL")
```

都是用同一个 `robot.data` 当前状态计算东西。

### 8.1 为什么 MPC/WBC 是 class

```python
mpc = CentroidalMPC(mpc_config)
stance_controller = StanceWBCQP(...)
```

因为它们不是一次性函数。它们长期持有：

```text
config
OSQP solver
last solution
problem shape
```

例如 WBC 里：

```python
self.config
self._solver
self._last_solution
```

这种对象状态放在 class 里更自然。

## 9. @property 是什么

`model_interface.py` 里有：

```python
@property
def nq(self) -> int:
    return self.model.nq
```

所以你可以写：

```python
robot.nq
```

而不是：

```python
robot.nq()
```

property 的意义：

```text
看起来像字段
实际是通过函数计算/返回
```

项目里：

```python
robot.q
robot.v
robot.nq
robot.nv
robot.nu
```

都可以当“只读属性”理解。

## 10. type hint 怎么读

例如：

```python
def solve(
    self,
    robot: MuJoCoModelInterface,
    qpos_ref: Array,
    force_ref: Array | None = None,
    stance_pos_refs: dict[str, Array] | None = None,
) -> StanceWBCSolution:
```

读法：

```text
robot 是 MuJoCoModelInterface
qpos_ref 是 numpy array
force_ref 可以是 numpy array，也可以是 None
stance_pos_refs 可以是 dict[str, Array]，也可以是 None
返回 StanceWBCSolution
```

`| None` 表示可选。

`dict[str, Array]` 表示：

```text
key 是 str，比如 "FL"
value 是 Array，比如 foot position
```

`tuple[str, ...]` 表示：

```text
一个字符串 tuple，长度不固定
```

例如：

```python
("FL", "RR")
("FR", "RL")
("FL", "FR", "RL", "RR")
```

## 11. 函数默认参数

例如：

```python
def solve(self, robot, qpos_ref, force_ref: Array | None = None):
```

调用时可以不传：

```python
solution = controller.solve(robot, qpos_ref)
```

函数内部会看到：

```python
force_ref is None
```

然后自己生成默认值。

也可以传：

```python
solution = controller.solve(robot, qpos_ref, force_ref=mpc_force_ref)
```

这里 `force_ref=...` 是 keyword argument，好处是清楚。

## 12. slice 怎么读

项目里大量使用 slice。

```python
qpos[0:3]     # base position
qpos[3:7]     # base quaternion
qpos[7:]      # 12 joint positions

qvel[0:3]     # base linear velocity
qvel[3:6]     # base angular velocity
qvel[6:]      # 12 joint velocities
```

`0:3` 表示取下标：

```text
0, 1, 2
```

不包含 3。

QP 里常见：

```python
idx_vdot = slice(0, nv)
idx_tau = slice(nv, nv + nu)
idx_force = slice(nv + nu, nvar)
```

这就是把一个大变量向量切成三段：

```text
z = [vdot, tau, f]
```

然后：

```python
vdot = z[idx_vdot]
tau = z[idx_tau]
contact_forces = z[idx_force]
```

## 13. reshape / vstack / concatenate

MPC/WBC 里经常需要把多个向量堆成一个大向量。

### 13.1 reshape

```python
fz = mpc_force_ref.reshape(len(FOOT_GEOMS), 3)[:, 2]
```

如果：

```text
mpc_force_ref shape = (12,)
```

代表：

```text
[fx_FL, fy_FL, fz_FL, fx_FR, fy_FR, fz_FR, ...]
```

reshape 后：

```text
shape = (4, 3)
每一行是一条腿的 [fx, fy, fz]
```

`[:, 2]` 表示：

```text
所有行，第 3 列
也就是四条腿的 fz
```

### 13.2 vstack

```python
np.vstack([forces_by_foot[foot] for foot in selected_feet])
```

意思：

```text
把多个 3D force 向量按行堆起来
```

### 13.3 concatenate

```python
l = np.concatenate([beq_dyn, beq_stance, l_friction, l_tau])
```

意思：

```text
把多个约束下界向量拼成 OSQP 的一个大 l
```

OSQP 标准形式：

```text
min 0.5 z^T P z + q^T z
s.t. l <= A z <= u
```

所以项目里：

```python
P / q
A / l / u
```

就是 QP 矩阵。

## 14. model_interface.py 怎么读

这个文件是 MuJoCo 和控制数学之间的桥。

核心对象：

```python
robot = MuJoCoModelInterface(MODEL_PATH)
```

它内部持有：

```text
robot.model   MuJoCo model
robot.data    MuJoCo current state
```

重点函数：

```text
mass_matrix()        -> M(q)
bias_forces()        -> h(q,v)
actuation_matrix()   -> B
geom_jacobian()      -> foot Jacobian
stacked_geom_jacobian()
stacked_geom_jdot_v()
geom_position()
geom_velocity()
center_of_mass()
composite_inertia_world_about_com()
base_rotation_world_from_base()
```

读法：

```text
WBC/MPC 不直接碰 MuJoCo 底层 API
它们通过 robot.mass_matrix(), robot.geom_position() 等接口拿数学量
```

最重要的方程：

```text
M(q) vdot + h(q,v) = B tau + Jc(q)^T f
```

`model_interface.py` 负责给出：

```text
M
h
B
Jc
Jdot*v
foot position
COM
inertia
```

## 15. wbc_qp.py 怎么读

WBC 是全身控制核心。

主线控制器：

```text
StanceWBCQP
SingleLegSwingWBCQP
GeneralContactWBCQP
```

现在 locomotion 主链路主要用：

```text
StanceWBCQP          四脚 stance
GeneralContactWBCQP  任意非腾空接触模式，例如 trot 两脚 stance 两脚 swing
```

### 15.1 WBC 决策变量

```text
z = [vdot, tau, f]
```

代码里：

```python
nv = robot.nv
nu = robot.nu
nf = 3 * len(stance_feet)
nvar = nv + nu + nf

idx_vdot = slice(0, nv)
idx_tau = slice(nv, nv + nu)
idx_force = slice(nv + nu, nvar)
```

物理意义：

```text
vdot   generalized acceleration
tau    12 个关节力矩
f      stance feet contact forces
```

### 15.2 WBC 约束

动力学硬约束：

```text
M vdot + h = B tau + Jc^T f
```

stance 足端加速度约束：

```text
Jc vdot + Jdot_c v = xddot_stance_cmd
```

摩擦锥：

```text
|fx| <= mu fz
|fy| <= mu fz
fz >= normal_force_min
```

力矩限制：

```text
tau_min <= tau <= tau_max
```

这些最后都被堆成：

```python
a = sparse.vstack([...])
l = np.concatenate([...])
u = np.concatenate([...])
```

送给 OSQP。

### 15.3 WBC cost

WBC 不是随便求一个满足动力学的解，它还要偏好某些行为：

```text
base position acceleration tracking
base orientation acceleration tracking
joint posture tracking
swing foot task tracking
tau regularization
force tracking / force regularization
```

在代码里体现为：

```python
p_diag = np.zeros(nvar)
q = np.zeros(nvar)
```

然后不断往 `p_diag` 和 `q` 里加项。

这里是 diagonal QP cost：

```text
0.5 z^T P z + q^T z
```

如果想让某段变量跟踪 `target`，代码会加类似：

```text
weight * ||x - target||^2
```

展开后进入：

```text
P 对角线
q 线性项
```

## 16. centroidal_mpc.py 怎么读

MPC 是上层接触力规划。

它不是完整机器人动力学，而是 single-rigid-body / centroidal 近似。

核心状态大致是：

```text
COM position
COM velocity
orientation small error
angular velocity
```

核心变量是 horizon 上每条腿的接触力：

```text
f_i[k]
```

MPC 输出：

```python
mpc_solution.first_contact_forces
```

这就是当前时刻 WBC 要跟踪的每足力参考。

典型调用：

```python
mpc_solution = mpc.solve(
    robot,
    com_ref,
    com_velocity_ref=com_vel_ref,
    orientation_ref=orientation_ref,
    angular_velocity_ref=angular_velocity_ref,
    contact_schedule=contact_schedule,
)
mpc_force_ref = mpc_solution.first_contact_forces
```

读法：

```text
给 MPC 当前机器人状态和未来接触模式
MPC 解一个 horizon QP
拿第一步接触力给 WBC
```

## 17. run_trot_reference_viewer.py 主循环

这是最值得读的入口脚本。

核心结构：

```python
while viewer.is_running():
    sim_time = float(robot.data.time)

    # 1. 更新 gait/contact schedule/swing reference/base reference
    # 2. 到 MPC 频率时，求接触力
    # 3. 到 WBC 频率时，求 tau
    # 4. 把 tau 写进 MuJoCo
    # 5. mj_step
    # 6. viewer.sync
```

关键变量：

```text
locked_positions    dict: foot -> stance foot position
active_plans        dict: foot -> SwingPlan
swing_refs          dict: foot -> SwingReference
generic_controllers dict: contact mode -> WBC controller
mpc_force_ref       ndarray: all feet contact force reference
last_tau            ndarray: held torque command
```

### 17.1 locked_positions

```python
initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}
```

意义：

```text
planner 认为 stance 脚应该锁在哪个世界坐标点
```

swing 落地后：

```python
locked_positions[foot] = active_plans[foot].target_position.copy()
```

意义：

```text
这只脚的新落点成为新的 stance 锁定点
```

### 17.2 active_plans

```python
active_plans: dict[str, SwingPlan] = {}
```

当一个 trot window 开始：

```python
active_plans = {
    foot: SwingPlan(
        foot=foot,
        start_position=locked_positions[foot].copy(),
        target_position=locked_positions[foot] + foothold_delta_for_foot(...),
    )
    for foot in window.swing_feet
}
```

意义：

```text
对当前 swing 的每条腿，记录起点和目标落点
```

### 17.3 swing_refs

```python
swing_refs = {
    foot: swing_foothold_reference(...)
    for foot, plan in active_plans.items()
}
```

意义：

```text
根据 SwingPlan 和当前时间，生成期望位置/速度/加速度
```

WBC 使用：

```python
swing_pos_refs={foot: ref.position for foot, ref in swing_refs.items()}
swing_vel_refs={foot: ref.velocity for foot, ref in swing_refs.items()}
swing_acc_refs={foot: ref.acceleration for foot, ref in swing_refs.items()}
```

意义：

```text
把 swing 轨迹拆成 WBC 需要的三个 dict
```

### 17.4 generic_controllers

```python
generic_controllers: dict[tuple[tuple[str, ...], tuple[str, ...]], GeneralContactWBCQP] = {}
```

key 是：

```python
key = (stance_feet, swing_feet)
```

如果不存在：

```python
if key not in generic_controllers:
    generic_controllers[key] = GeneralContactWBCQP(...)
```

意义：

```text
不同 contact mode 的 WBC 配置不同
第一次遇到某个 mode 时创建
之后复用
```

## 18. record_trot_demo.py 怎么读

它和 `run_trot_reference_viewer.py` 很像，但目的不同。

```text
run_trot_reference_viewer.py
    实时边算边显示

record_trot_demo.py
    先离线 rollout 保存状态，再生成 GIF 或 viewer replay
```

为什么有它？

```text
Python QP 求解慢，实时 viewer 可能卡
离线 rollout 后按固定帧率播放，展示更平滑
```

你先不用逐行读它。只要知道它复用了同样主链路：

```text
MuJoCo state
-> contact schedule / reference
-> MPC force
-> WBC tau
-> mj_step
-> save qpos/qvel
```

## 19. OSQP 在代码里的形状

OSQP 求解形式：

```text
min 0.5 z^T P z + q^T z
s.t. l <= A z <= u
```

代码里变量名固定：

```text
p / q / a / l / u
```

注意：

```python
p = sparse.diags(...)
a = sparse.vstack(...)
solver.setup(P=p, q=q, A=a, l=l, u=u)
result = solver.solve()
```

`P` 是二次项矩阵，不是 position。

`q` 是线性项，不是 generalized coordinate。

这点容易混。

## 20. 常见 Python 语法对照表

```python
if x is None:
```

表示：

```text
如果 x 没传/没有值
```

```python
if status in ("solved", "solved inaccurate"):
```

表示：

```text
status 是这两个字符串之一
```

```python
tuple(foot for foot in FOOT_GEOMS if foot not in swing_feet)
```

表示：

```text
从四条腿里选出不在 swing_feet 里的腿，作为 stance_feet
```

```python
np.zeros(robot.nu)
```

生成全 0 向量。

```python
np.asarray(x, dtype=float)
```

把输入转成 numpy float 数组。

```python
np.linalg.norm(x)
```

求向量范数。

```python
np.clip(x, -1.0, 1.0)
```

把 x 限制在区间内。

```python
float(...)
```

转成 Python 标量 float，常用于打印或日志。

```python
getattr(solution, "swing_accel_error", np.zeros(0))
```

表示：

```text
如果 solution 有 swing_accel_error 字段，就取它
否则返回 np.zeros(0)
```

因为 stance solution 没有 swing error，而 swing solution 有。

## 21. 你读代码时的检查清单

每读一个函数，写下：

```text
函数名：
输入：
输出：
修改了哪些对象：
有没有调用 MuJoCo：
有没有调用 OSQP：
物理意义：
```

例如：

```text
函数：GeneralContactWBCQP.solve
输入：robot, qpos_ref, swing refs, force_ref, stance_pos_refs
输出：GeneralContactWBCSolution
修改：可能更新 self._solver / self._last_solution
MuJoCo：通过 robot 接口读 M, h, J
OSQP：是
物理意义：在完整动力学约束下求 vdot/tau/contact force
```

## 22. 第一轮阅读任务

不要试图一次读完所有文件。按这个节奏：

### 第 1 轮：只读框架

```text
README.md
docs/mainline_architecture.md
docs/control_stack.md
docs/project_structure.md
```

目标：

```text
能画出 planner -> MPC -> WBC -> MuJoCo
```

### 第 2 轮：读 MuJoCo 接口

```text
conventions.py
model_interface.py
```

目标：

```text
知道 qpos/qvel 怎么约定
知道 M, h, B, J, Jdot*v 从哪里来
```

### 第 3 轮：读 WBC

```text
wbc_qp.py
```

目标：

```text
能说清楚 z = [vdot, tau, f]
能说清楚 dynamics / stance / swing / friction / torque constraints
能看懂 P, q, A, l, u 怎么拼
```

### 第 4 轮：读 MPC

```text
centroidal_mpc.py
```

目标：

```text
知道 MPC 用 SRB/centroidal 模型
知道 contact_schedule 如何让某些脚 force = 0
知道 first_contact_forces 怎么给 WBC
```

### 第 5 轮：读主脚本

```text
run_trot_reference_viewer.py
record_trot_demo.py
```

目标：

```text
知道每个仿真步发生什么
知道 dict 如何保存 foot -> reference / position / controller
知道 MPC/WBC 频率和 last_tau hold
```

## 23. 面试讲法

如果要用一句话讲代码结构：

```text
我把 MuJoCo 的动力学/运动学封装在 model_interface.py，
上层脚本生成 gait/contact schedule 和 swing/base reference，
centroidal_mpc.py 用 SRB 模型优化每条腿的接触力，
wbc_qp.py 在完整 floating-base 动力学约束下求 [vdot, tau, f]，
最后把 tau 写入 MuJoCo actuator ctrl。
```

这就是当前项目代码的主线。

## 24. 只看手册版：核心代码带注释

这一章把当前项目最关键的代码逻辑摘出来并加解释。

注意：这里不是逐字复制所有 `.py` 文件。完整代码有几千行，如果全部贴进来，反而会淹没主线。这里保留的是你理解项目、讲项目、以后迁移到 C++/ROS2 最需要的代码骨架。

你可以先只看这一章，不打开 `.py` 文件。

## 25. MuJoCoModelInterface：机器人模型接口

这个类在：

```text
src/mujoco_wbc/model_interface.py
```

它的作用是：

```text
把 MuJoCo 的底层 model/data 封装成控制器需要的数学接口
```

核心代码骨架：

```python
class MuJoCoModelInterface:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.base_body_name = BASE_BODY_NAME
        self.base_body_id = self._body_id(self.base_body_name)
        mujoco.mj_forward(self.model, self.data)
```

逐句解释：

```text
self.model_path
    保存模型路径

self.model
    MuJoCo 的 MjModel，包含机器人结构、质量、关节、actuator、geom 等静态信息

self.data
    MuJoCo 的 MjData，包含当前 qpos/qvel/contact/force 等动态状态

self.base_body_id
    找到 floating base body 的 id，后面读 base pose、COM、inertia 会用

mj_forward
    根据当前 qpos/qvel 刷新 MuJoCo 派生量，例如 xpos、xmat、subtree_com
```

几个 property：

```python
@property
def nq(self) -> int:
    return self.model.nq

@property
def nv(self) -> int:
    return self.model.nv

@property
def nu(self) -> int:
    return self.model.nu

@property
def q(self) -> Array:
    return self.data.qpos.copy()

@property
def v(self) -> Array:
    return self.data.qvel.copy()
```

读法：

```text
robot.nq   generalized coordinate 维度，Go2 是 19
robot.nv   generalized velocity 维度，Go2 是 18
robot.nu   actuator/control 维度，Go2 是 12
robot.q    当前 qpos 的 copy
robot.v    当前 qvel 的 copy
```

为什么 `q` 和 `v` 返回 `.copy()`？

```text
因为外部代码拿到 q/v 后通常只是做参考或计算，不应该无意中改 MuJoCo data。
```

### 25.1 M(q)

```python
def mass_matrix(self) -> Array:
    mass = np.zeros((self.nv, self.nv), dtype=float)
    mujoco.mj_fullM(self.model, self.data, mass)
    return mass
```

物理意义：

```text
返回 generalized velocity space 下的质量矩阵 M(q)
shape = (18, 18)
```

WBC 会用它构造：

```text
M vdot + h = B tau + J^T f
```

### 25.2 h(q,v)

```python
def bias_forces(self, include_passive: bool = False) -> Array:
    if include_passive:
        return self.data.qfrc_bias.copy()
    return self.data.qfrc_bias.copy() - self.data.qfrc_passive.copy()
```

MuJoCo 里：

```text
qfrc_bias
    gravity / coriolis / centrifugal

qfrc_passive
    damping / friction / spring 等 passive force
```

项目默认：

```text
h = qfrc_bias - qfrc_passive
```

因为 WBC 方程写成：

```text
M vdot + h = B tau + J^T f
```

passive generalized force 如果本来在右边，就要移到左边。

### 25.3 B 矩阵

```python
def actuation_matrix(self) -> Array:
    original_ctrl = self.data.ctrl.copy()
    original_qacc = self.data.qacc.copy()
    matrix = np.zeros((self.nv, self.nu), dtype=float)

    self.data.ctrl[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

    for actuator_id in range(self.nu):
        self.data.ctrl[:] = 0.0
        self.data.ctrl[actuator_id] = 1.0
        mujoco.mj_forward(self.model, self.data)
        matrix[:, actuator_id] = self.data.qfrc_actuator

    self.data.ctrl[:] = original_ctrl
    self.data.qacc[:] = original_qacc
    mujoco.mj_forward(self.model, self.data)
    return matrix
```

物理意义：

```text
B 是 actuator torque 到 generalized force 的映射
qfrc_actuator = B tau
```

为什么一列一列算？

```text
把第 i 个 actuator control 设成 1，其它设 0
MuJoCo 给出这时候的 qfrc_actuator
这就是 B 的第 i 列
```

为什么最后恢复 `ctrl/qacc`？

```text
因为这个函数只是读 B，不应该改变仿真状态。
```

### 25.4 足端位置

```python
def geom_position(self, geom_name: str) -> Array:
    geom_id = self._geom_id(geom_name)
    center = self.data.geom_xpos[geom_id].copy()
    if self._uses_foot_contact_point(geom_name, geom_id):
        radius = float(self.model.geom_size[geom_id, 0])
        return center - np.array([0.0, 0.0, radius], dtype=float)
    return center
```

重点：

```text
Go2 foot geom 是球
MuJoCo 的 geom_xpos 是球心
但控制里更想用“球底部接触点”
所以足端位置 = center - [0, 0, radius]
```

这和之前你观察到的“脚实际会陷进地面一点”有关。

### 25.5 足端 Jacobian

```python
def geom_jacobian(self, geom_name: str) -> FrameJacobian:
    geom_id = self._geom_id(geom_name)
    jacp = np.zeros((3, self.nv), dtype=float)
    jacr = np.zeros((3, self.nv), dtype=float)
    mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, geom_id)
    return FrameJacobian(jacp=jacp, jacr=jacr)
```

返回：

```text
jacp: translational Jacobian, shape = (3, 18)
jacr: rotational Jacobian, shape = (3, 18)
```

WBC 主要用足端平移 Jacobian：

```text
foot velocity = Jp v
foot acceleration = Jp vdot + Jdot v
```

### 25.6 多足 Jacobian 堆叠

项目里会把多个脚的 Jacobian 堆起来：

```python
jc = robot.stacked_geom_jacobian(list(stance_feet))
```

物理意义：

```text
如果 stance_feet = ("FR", "RL")
那么 Jc shape = (6, 18)
前 3 行是 FR
后 3 行是 RL
```

这对应：

```text
Jc vdot + Jdot_c v = 0
```

## 26. WBC：StanceWBCQP.solve 骨架

这个文件在：

```text
src/mujoco_wbc/wbc_qp.py
```

WBC 的核心变量：

```text
z = [vdot, tau, f]
```

对应代码：

```python
nv = robot.nv
nu = robot.nu
nf = 3 * len(cfg.foot_geoms)
nvar = nv + nu + nf

idx_vdot = slice(0, nv)
idx_tau = slice(nv, nv + nu)
idx_force = slice(nv + nu, nvar)
```

解释：

```text
nv = 18
nu = 12
nf = 3 * stance_foot_count

idx_vdot  用来从 z 里取 vdot
idx_tau   用来从 z 里取 tau
idx_force 用来从 z 里取 contact force
```

### 26.1 读取动力学量

```python
mass = robot.mass_matrix()
h = robot.bias_forces()
bmat = robot.actuation_matrix()
jc = robot.stacked_geom_jacobian(list(cfg.foot_geoms))
jdot_v = robot.stacked_geom_jdot_v(list(cfg.foot_geoms))
```

物理意义：

```text
mass = M(q)
h    = h(q,v)
bmat = B
jc   = stance contact Jacobian
jdot_v = Jdot_c v
```

这些来自 MuJoCo，但 WBC 不直接调用 MuJoCo API，而是通过 `robot` 接口拿。

### 26.2 生成任务加速度

```python
pos_acc_cmd = self._base_position_accel_cmd(robot, qpos_ref)
ori_acc_cmd = self._base_orientation_accel_cmd(robot, qpos_ref)
joint_acc_cmd = self._joint_accel_cmd(robot, qpos_ref)
stance_acc_cmd = self._stance_accel_cmd(robot, list(cfg.foot_geoms), stance_pos_refs)
```

这些都是 PD 形式的加速度命令，例如：

```text
xddot_cmd = kp * (x_ref - x) + kd * (xd_ref - xd)
```

区别：

```text
base_position_accel_cmd     想让 base 位置跟踪 qpos_ref[0:3]
base_orientation_accel_cmd  想让 base 姿态跟踪 qpos_ref[3:7]
joint_accel_cmd             想让关节姿态接近 home/reference
stance_accel_cmd            想让 stance foot 锁在 stance_pos_refs
```

### 26.3 构造 cost

代码骨架：

```python
p_diag = np.zeros(nvar)
q = np.zeros(nvar)

self._add_diagonal_tracking_cost(
    p_diag, q, slice(0, 3), cfg.weight_base_pos, pos_acc_cmd
)
self._add_diagonal_tracking_cost(
    p_diag, q, slice(3, 6), cfg.weight_base_ori, ori_acc_cmd
)
self._add_diagonal_tracking_cost(
    p_diag, q, slice(6, nv), cfg.weight_joint_posture, joint_acc_cmd
)

p_diag[idx_tau] += cfg.weight_tau
p_diag[idx_force] += cfg.weight_force + force_zero_weights
q[idx_force] += -cfg.weight_force * force_ref

p = sparse.diags(p_diag + 1.0e-9, format="csc")
```

这段在做：

```text
base acceleration tracking
orientation acceleration tracking
joint posture tracking
torque regularization
contact force tracking / regularization
```

为什么是 `p_diag`？

```text
当前 WBC 的 cost 是 diagonal 二次项
所以只需要维护 P 的对角线
```

`_add_diagonal_tracking_cost` 可以理解成：

```text
给某段变量 x 加 weight * ||x - target||^2
```

展开后进入：

```text
P 对角线增加 weight
q 线性项增加 -weight * target
```

### 26.4 动力学等式约束

代码：

```python
aeq_dyn = sparse.hstack(
    [sparse.csc_matrix(mass), sparse.csc_matrix(-bmat), sparse.csc_matrix(-jc.T)],
    format="csc",
)
beq_dyn = -h
```

它对应：

```text
M vdot + h = B tau + Jc^T f
```

移项：

```text
M vdot - B tau - Jc^T f = -h
```

所以矩阵块是：

```text
[M, -B, -Jc^T] [vdot, tau, f] = -h
```

### 26.5 stance 足端约束

代码：

```python
aeq_stance = sparse.hstack(
    [
        sparse.csc_matrix(jc),
        sparse.csc_matrix((nf, nu)),
        sparse.csc_matrix((nf, nf)),
    ],
    format="csc",
)
beq_stance = stance_acc_cmd - jdot_v
```

对应：

```text
Jc vdot + Jdot_c v = stance_acc_cmd
```

移项：

```text
Jc vdot = stance_acc_cmd - Jdot_c v
```

所以矩阵块是：

```text
[Jc, 0, 0] [vdot, tau, f] = stance_acc_cmd - jdot_v
```

### 26.6 摩擦锥和力矩限制

代码：

```python
a_friction, l_friction, u_friction = self._friction_constraints(
    nv, nu, nf, cfg.friction_mu, cfg.normal_force_min
)
a_tau, l_tau, u_tau = self._torque_constraints(robot, nv, nu, nf)
```

物理意义：

```text
friction:
    |fx| <= mu fz
    |fy| <= mu fz
    fz >= normal_force_min

torque:
    tau_min <= tau <= tau_max
```

### 26.7 拼成 OSQP

代码：

```python
a = sparse.vstack([aeq_dyn, aeq_stance, a_friction, a_tau], format="csc")
l = np.concatenate([beq_dyn, beq_stance, l_friction, l_tau])
u = np.concatenate([beq_dyn, beq_stance, u_friction, u_tau])

result = self._solve_osqp(p, q, a, l, u)
```

OSQP 标准形式：

```text
min 0.5 z^T P z + q^T z
s.t. l <= A z <= u
```

对于等式约束：

```text
l = u = b
```

所以 `beq_dyn` 和 `beq_stance` 同时进入 l/u。

### 26.8 拆解解向量

代码：

```python
z = result.x if result.x is not None else np.zeros(nvar)

vdot = z[idx_vdot]
tau = z[idx_tau]
contact_forces = z[idx_force]
```

物理意义：

```text
OSQP 求的是一个大向量 z
WBC 从里面切出 vdot, tau, f
```

然后算 residual：

```python
dynamics_residual = mass @ vdot + h - bmat @ tau - jc.T @ contact_forces
stance_residual = jc @ vdot + jdot_v - stance_acc_cmd
```

用于检查：

```text
动力学约束是否满足
stance 足端约束是否满足
```

## 27. GeneralContactWBCQP：trot/crawl 主力 WBC

`StanceWBCQP` 只处理四脚 stance。

真正 locomotion 用的是：

```text
GeneralContactWBCQP
```

它支持：

```text
3 stance + 1 swing    crawl
2 stance + 2 swing    trot
4 stance              stance
```

它的输入多了 swing refs：

```python
solution = generic_controller.solve(
    robot,
    base_ref,
    swing_pos_refs={foot: ref.position for foot, ref in swing_refs.items()},
    swing_vel_refs={foot: ref.velocity for foot, ref in swing_refs.items()},
    swing_acc_refs={foot: ref.acceleration for foot, ref in swing_refs.items()},
    force_ref=force_ref_for_feet(mpc_force_ref, stance_feet),
    stance_pos_refs={foot: locked_positions[foot] for foot in stance_feet},
)
```

读法：

```text
robot
    当前 MuJoCo 状态

base_ref
    期望 base qpos

swing_pos_refs / swing_vel_refs / swing_acc_refs
    每条 swing 腿的足端期望轨迹

force_ref
    MPC 给 stance feet 的接触力参考

stance_pos_refs
    stance feet 应锁定的位置
```

### 27.1 swing task 的物理意义

WBC 不做 IK，而是写成任务空间加速度：

```text
J_sw vdot + Jdot_sw v = xddot_cmd
```

其中：

```text
xddot_cmd =
    swing_acc_ref
    + kp_swing * (swing_pos_ref - swing_pos)
    + kd_swing * (swing_vel_ref - swing_vel)
```

这个任务作为 soft cost 进入 QP。

所以：

```text
stance feet 是硬约束
swing feet 是软任务
```

这就是 WBC 比 IK 更自然的地方：它直接在 full-body dynamics 里找 torque。

## 28. CentroidalMPC：接触力规划

这个文件在：

```text
src/mujoco_wbc/centroidal_mpc.py
```

当前主线用的是：

```python
mpc = CentroidalMPC(mpc_config)
mpc_solution = mpc.solve(
    robot,
    com_ref,
    com_velocity_ref=com_vel_ref,
    orientation_ref=orientation_ref,
    angular_velocity_ref=angular_velocity_ref,
    contact_schedule=contact_schedule,
)
mpc_force_ref = mpc_solution.first_contact_forces
```

读法：

```text
给 MPC 当前 robot 状态
给未来 COM/reference
给未来 contact schedule
MPC 输出第一步每足接触力
```

### 28.1 MPC 配置

```python
mpc_config = CentroidalMPCConfig(
    contact_geoms=FOOT_GEOMS,
    horizon_steps=12,
    dt=0.03,
    normal_force_min=5.0,
    weight_orientation=1200.0,
    weight_angular_velocity=100.0,
)
```

解释：

```text
horizon_steps=12
    预测 12 个 knot

dt=0.03
    每个 knot 间隔 0.03s

contact_geoms
    四条腿顺序

weight_orientation / weight_angular_velocity
    姿态和角速度跟踪权重
```

### 28.2 contact_schedule

MPC 的 contact schedule 是布尔矩阵：

```text
shape = (horizon_steps, 4)
```

例如：

```text
True  = stance, 允许该脚产生接触力
False = swing, 该脚接触力约束为 0
```

代码里生成：

```python
contact_schedule = trot_contact_schedule(
    windows,
    sim_time,
    mpc_config.horizon_steps,
    mpc_config.dt,
    active_window=current_window,
)
```

### 28.3 MPC 输出给 WBC

```python
mpc_force_ref = mpc_solution.first_contact_forces
```

shape：

```text
(12,)
```

排列：

```text
[fx_FL, fy_FL, fz_FL,
 fx_FR, fy_FR, fz_FR,
 fx_RL, fy_RL, fz_RL,
 fx_RR, fy_RR, fz_RR]
```

如果当前 WBC 只有一部分 stance feet，需要筛选：

```python
force_ref_for_feet(mpc_force_ref, stance_feet)
```

这会从四脚力里选出 stance feet 对应的力。

## 29. run_trot_reference_viewer.py：完整主循环代码骨架

这是实时 viewer demo 的主入口。

先初始化：

```python
robot = MuJoCoModelInterface(MODEL_PATH)
robot.set_keyframe("home")

home_qpos_ref = robot.q.copy()
home_com_ref = robot.center_of_mass()
initial_base_pos = robot.data.qpos[0:3].copy()
initial_foot_positions = {foot: robot.geom_position(foot) for foot in FOOT_GEOMS}
locked_positions = {foot: pos.copy() for foot, pos in initial_foot_positions.items()}
```

解释：

```text
robot
    MuJoCo robot wrapper

home_qpos_ref
    初始 home 姿态，后面 base/joint ref 会从它改

home_com_ref
    初始整机 COM

initial_base_pos
    初始 base 位置

initial_foot_positions
    foot -> 初始足端位置

locked_positions
    foot -> planner 认为 stance 脚应锁定的位置
```

创建 MPC/WBC：

```python
mpc = CentroidalMPC(mpc_config)
stance_controller = StanceWBCQP(...)
generic_controllers: dict[tuple[tuple[str, ...], tuple[str, ...]], GeneralContactWBCQP] = {}
```

解释：

```text
mpc
    每隔 mpc_dt 求一次接触力

stance_controller
    四脚 stance 时用

generic_controllers
    不同 contact mode 对应不同 GeneralContactWBCQP，对象缓存起来复用
```

主循环骨架：

```python
while viewer.is_running():
    sim_time = float(robot.data.time)

    # 1. 根据时间决定是否进入新的 swing window
    # 2. 如果 swing 结束，则把 swing target 写入 locked_positions
    # 3. 生成 swing_refs
    # 4. 生成 contact_schedule
    # 5. 生成 base_ref / com_ref / orientation_ref
    # 6. 到 MPC 频率时，解 MPC，更新 mpc_force_ref
    # 7. 到 WBC 频率时，解 WBC，更新 last_tau
    # 8. robot.data.ctrl[:] = last_tau
    # 9. mujoco.mj_step(robot.model, robot.data)
    # 10. viewer.sync()
```

### 29.1 进入 swing window

```python
if active_window_id is None and next_window_id < len(windows) and sim_time >= windows[next_window_id].start_time:
    active_window_id = next_window_id
    window = windows[active_window_id]
    active_plans = {
        foot: SwingPlan(
            foot=foot,
            start_position=locked_positions[foot].copy(),
            target_position=locked_positions[foot] + foothold_delta_for_foot(...),
        )
        for foot in window.swing_feet
    }
    next_wbc_update = sim_time
    next_mpc_update = sim_time
```

读法：

```text
如果当前没有 swing，并且时间到了下一个 swing window：
    激活这个 window
    给每条 swing 腿生成 SwingPlan
    强制 MPC/WBC 马上刷新
```

### 29.2 swing 落地结束

```python
if current_window is not None and should_finish_trot_window(...):
    for foot in current_window.swing_feet:
        locked_positions[foot] = active_plans[foot].target_position.copy()
    active_window_id = None
    next_window_id += 1
    active_plans = {}
    current_window = None
    next_wbc_update = sim_time
    next_mpc_update = sim_time
```

读法：

```text
swing 结束后：
    目标落点变成新的 stance 锁定点
    清空 active_plans
    进入 stance 或下一个 window
    强制 MPC/WBC 用新的 contact mode 刷新
```

### 29.3 生成 swing_refs

```python
swing_refs = {}
if current_window is not None:
    swing_refs = {
        foot: swing_foothold_reference(
            initial_position=plan.start_position,
            step_delta=plan.target_position - plan.start_position,
            swing_height=args.swing_height,
            start_time=current_window.start_time,
            duration=current_window.duration,
            time_s=sim_time,
        )
        for foot, plan in active_plans.items()
    }
```

读法：

```text
如果当前有 swing 腿：
    对每条 swing 腿，根据起点、终点、抬脚高度、当前时间生成轨迹点
```

`swing_foothold_reference` 输出：

```text
position
velocity
acceleration
```

### 29.4 生成 reference

```python
contact_schedule = trot_contact_schedule(...)
planned_foot_positions = planned_feet_from_refs(locked_positions, swing_refs)
yaw_ref = args.yaw_rate * command_time

base_ref = foot_centered_base_reference(
    home_qpos_ref,
    initial_base_pos,
    initial_foot_positions,
    planned_foot_positions,
    yaw=yaw_ref,
)

com_ref = home_com_ref.copy()
com_ref[0:2] += base_ref[0:2] - initial_base_pos[0:2]

com_vel_ref = np.array([args.vx, args.vy, 0.0], dtype=float)
orientation_ref = np.array([0.0, 0.0, yaw_ref], dtype=float)
angular_velocity_ref = np.array([0.0, 0.0, args.yaw_rate], dtype=float)
```

读法：

```text
contact_schedule
    未来 horizon 每个脚是 stance 还是 swing

planned_foot_positions
    当前 planner 认为每条腿在哪里

base_ref
    根据足端平均位置生成 base 参考

com_ref
    用 base_ref 的 xy 位移近似整机 COM 参考

orientation_ref
    roll/pitch/yaw 参考，这里主要是 yaw
```

### 29.5 MPC 更新

```python
if sim_time >= next_mpc_update:
    mpc_solution = mpc.solve(
        robot,
        com_ref,
        com_velocity_ref=com_vel_ref,
        orientation_ref=orientation_ref,
        angular_velocity_ref=angular_velocity_ref,
        contact_schedule=contact_schedule,
    )
    mpc_force_ref = mpc_solution.first_contact_forces
    next_mpc_update += args.mpc_dt
```

读法：

```text
MPC 不一定每个 MuJoCo timestep 都算
到时间才算一次
两次之间保持上一拍 mpc_force_ref
```

### 29.6 WBC 更新

四脚 stance：

```python
if current_window is None:
    solution = stance_controller.solve(
        robot,
        base_ref,
        force_ref=mpc_force_ref,
        stance_pos_refs=locked_positions,
    )
```

读法：

```text
所有脚都是 stance
用 StanceWBCQP
四个脚都锁在 locked_positions
力参考来自 MPC 的四脚力
```

有 swing 腿：

```python
else:
    swing_feet = current_window.swing_feet
    stance_feet = tuple(foot for foot in FOOT_GEOMS if foot not in swing_feet)
    key = (stance_feet, swing_feet)

    if key not in generic_controllers:
        generic_controllers[key] = GeneralContactWBCQP(...)

    solution = generic_controllers[key].solve(
        robot,
        base_ref,
        swing_pos_refs={foot: ref.position for foot, ref in swing_refs.items()},
        swing_vel_refs={foot: ref.velocity for foot, ref in swing_refs.items()},
        swing_acc_refs={foot: ref.acceleration for foot, ref in swing_refs.items()},
        force_ref=force_ref_for_feet(mpc_force_ref, stance_feet),
        stance_pos_refs={foot: locked_positions[foot] for foot in stance_feet},
    )
```

读法：

```text
当前有 swing 腿
stance_feet = 不在 swing_feet 里的脚
用 GeneralContactWBCQP
stance 脚做硬约束
swing 脚跟踪 swing_refs
force_ref 只给 stance feet
```

### 29.7 写 torque 并仿真

```python
if solution.status in ("solved", "solved inaccurate"):
    last_tau = solution.tau.copy()
else:
    solve_failures += 1

robot.data.ctrl[:] = last_tau
mujoco.mj_step(robot.model, robot.data)
```

读法：

```text
如果 WBC 成功，更新 last_tau
如果失败，继续保持上一拍 last_tau
把 last_tau 写进 MuJoCo actuator ctrl
推进一步仿真
```

这就是完整闭环：

```text
state -> references -> MPC -> WBC -> tau -> MuJoCo step
```

## 30. contact_schedule 代码

核心函数：

```python
def trot_contact_schedule(
    windows: list[TrotWindow],
    current_time: float,
    horizon_steps: int,
    dt: float,
    active_window: TrotWindow | None = None,
) -> np.ndarray:
    schedule = np.ones((horizon_steps, len(FOOT_GEOMS)), dtype=bool)
    foot_to_index = {foot: idx for idx, foot in enumerate(FOOT_GEOMS)}
    for step in range(horizon_steps):
        knot_time = current_time + step * dt
        for window in windows:
            if window.start_time <= knot_time < window.end_time:
                for foot in window.swing_feet:
                    schedule[step, foot_to_index[foot]] = False
        if active_window is not None:
            for foot in active_window.swing_feet:
                schedule[step, foot_to_index[foot]] = False
    return schedule
```

逐句解释：

```text
schedule 初始全 True
    默认所有脚 stance

foot_to_index
    把 "FL" 映射到 0，"FR" 映射到 1...

for step in horizon
    对未来每个预测 knot 计算 knot_time

如果某个 knot_time 落在 swing window 内
    对应 swing feet 设成 False

active_window 额外强制当前 swing feet 为 False
    避免当前真实 WBC 已经切换，但 horizon 第一个 knot 还没覆盖到
```

最终：

```text
True  -> stance, MPC 可以给这个脚分配力
False -> swing, MPC 约束这个脚 force = 0
```

## 31. force_ref_for_feet 代码

```python
def force_ref_for_feet(force_ref_all: np.ndarray, selected_feet: tuple[str, ...]) -> np.ndarray:
    forces_by_foot = {
        foot: force
        for foot, force in zip(FOOT_GEOMS, force_ref_all.reshape(len(FOOT_GEOMS), 3))
    }
    return np.vstack([forces_by_foot[foot] for foot in selected_feet]).reshape(-1)
```

假设：

```text
force_ref_all shape = (12,)
```

先 reshape：

```text
shape = (4, 3)
每一行对应一条腿的 [fx, fy, fz]
```

然后：

```python
forces_by_foot = {
    "FL": force_FL,
    "FR": force_FR,
    "RL": force_RL,
    "RR": force_RR,
}
```

最后根据 `selected_feet` 取出需要的腿。

例如：

```python
selected_feet = ("FR", "RL")
```

返回：

```text
[fx_FR, fy_FR, fz_FR, fx_RL, fy_RL, fz_RL]
```

这个函数是 MPC/WBC 接口里的小胶水：

```text
MPC 总是输出四条腿的力
WBC 当前只需要 stance feet 的力
```

## 32. record_trot_demo.py 与 viewer 的区别

`record_trot_demo.py` 的核心 rollout 和 viewer 版本一样，但它不实时显示，而是保存：

```python
times: list[float] = []
qpos_samples: list[np.ndarray] = []
qvel_samples: list[np.ndarray] = []
```

每隔固定视觉采样时间：

```python
while robot.data.time >= next_sample_time:
    times.append(float(robot.data.time))
    qpos_samples.append(robot.data.qpos.copy())
    qvel_samples.append(robot.data.qvel.copy())
    next_sample_time += sample_dt
```

意思：

```text
控制器可以慢慢算
但只保存固定 fps 的状态
后面用这些状态平滑回放或渲染 GIF
```

所以它适合 GitHub 展示。

## 33. 读完手册后你应该能复述的版本

你不看 `.py` 文件，也应该能讲出：

```text
1. MuJoCoModelInterface 从 MuJoCo 拿 M, h, B, J, foot pos, COM。

2. run_trot_reference_viewer.py 维护 locked_positions / active_plans / swing_refs，
   根据当前时间生成 contact_schedule 和 references。

3. CentroidalMPC 根据 COM/orientation reference 和 contact_schedule，
   在 SRB 模型下求 horizon 上每条腿的 contact force。

4. WBC 的决策变量是 z = [vdot, tau, f]。
   它用完整 floating-base dynamics 作为硬约束：
       M vdot + h = B tau + Jc^T f
   用 stance acceleration 作为硬约束：
       Jc vdot + Jdot v = stance_acc_cmd
   用 swing foot task 作为 soft cost。

5. WBC 输出 tau，脚本写入 robot.data.ctrl[:]，
   然后 mujoco.mj_step 推进一步仿真。
```

这就是当前项目的代码主线。
