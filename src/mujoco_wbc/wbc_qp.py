"""First-pass stance WBC QP for a floating-base quadruped."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import osqp
from scipy import sparse

from .model_interface import MuJoCoModelInterface


Array = np.ndarray


class _WBCSolveProfiler:
    """Per-solve timing accumulator used by WBC controllers."""

    def __init__(self) -> None:
        self._sections_s: dict[str, float] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            self._sections_s[name] = self._sections_s.get(name, 0.0) + perf_counter() - start

    def milliseconds(self, total_start: float) -> dict[str, float]:
        profile = {name: 1000.0 * value for name, value in self._sections_s.items()}
        profile["total_solve"] = 1000.0 * (perf_counter() - total_start)
        return profile


@contextmanager
def _maybe_profile(profiler: _WBCSolveProfiler | None, name: str) -> Iterator[None]:
    if profiler is None:
        yield
        return
    with profiler.time(name):
        yield


@dataclass(frozen=True)
class StanceWBCConfig:
    foot_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    friction_mu: float = 0.6
    normal_force_min: float = 0.0
    weight_base_pos: float = 100.0
    weight_base_ori: float = 100.0
    weight_joint_posture: float = 10.0
    weight_tau: float = 1.0e-4
    weight_force: float = 1.0e-3
    kp_base_pos: float = 100.0
    kd_base_pos: float = 20.0
    kp_base_ori: float = 100.0
    kd_base_ori: float = 20.0
    kp_joint: float = 80.0
    kd_joint: float = 8.0
    kp_stance: float = 0.0
    kd_stance: float = 0.0
    use_jdot_v: bool = True


@dataclass(frozen=True)
class StanceWBCSolution:
    status: str
    vdot: Array
    tau: Array
    contact_forces: Array
    dynamics_residual: Array
    stance_residual: Array
    objective: float


@dataclass(frozen=True)
class SingleLegSwingWBCConfig:
    stance_foot_geoms: tuple[str, ...] = ("FR", "RL", "RR")
    swing_foot_geom: str = "FL"
    friction_mu: float = 0.6
    normal_force_min: float = 5.0
    weight_base_pos: float = 120.0
    weight_base_ori: float = 120.0
    weight_joint_posture: float = 6.0
    weight_swing_foot: float = 800.0
    weight_tau: float = 1.0e-4
    weight_force: float = 1.0e-3
    kp_base_pos: float = 120.0
    kd_base_pos: float = 24.0
    kp_base_ori: float = 120.0
    kd_base_ori: float = 24.0
    kp_joint: float = 40.0
    kd_joint: float = 5.0
    kp_swing: float = 400.0
    kd_swing: float = 40.0
    kp_stance: float = 0.0
    kd_stance: float = 0.0
    use_jdot_v: bool = True


@dataclass(frozen=True)
class SingleLegSwingWBCSolution(StanceWBCSolution):
    swing_accel_error: Array


@dataclass(frozen=True)
class GeneralContactWBCConfig:
    stance_foot_geoms: tuple[str, ...] = ("FR", "RL")
    swing_foot_geoms: tuple[str, ...] = ("FL", "RR")
    friction_mu: float = 0.6
    normal_force_min: float = 5.0
    weight_base_pos: float = 120.0
    weight_base_ori: float = 120.0
    weight_joint_posture: float = 6.0
    weight_swing_foot: float = 800.0
    weight_tau: float = 1.0e-4
    weight_force: float = 1.0e-3
    kp_base_pos: float = 120.0
    kd_base_pos: float = 24.0
    kp_base_ori: float = 120.0
    kd_base_ori: float = 24.0
    kp_joint: float = 40.0
    kd_joint: float = 5.0
    kp_swing: float = 400.0
    kd_swing: float = 40.0
    kp_stance: float = 0.0
    kd_stance: float = 0.0
    use_jdot_v: bool = True


@dataclass(frozen=True)
class GeneralContactWBCSolution(StanceWBCSolution):
    swing_accel_error: Array
    stance_foot_geoms: tuple[str, ...]
    swing_foot_geoms: tuple[str, ...]


class StanceWBCQP:
    """Solve one acceleration-level stance WBC QP.

    Decision variable:

        z = [vdot, tau, f]

    Hard constraints:

        M vdot + h = B tau + Jc^T f
        Jc vdot + Jdot_c v = 0
        |fx| <= mu fz, |fy| <= mu fz, fz >= 0
        tau_min <= tau <= tau_max
    """

    def __init__(self, config: StanceWBCConfig | None = None):
        self.config = config or StanceWBCConfig()
        self._solver: osqp.OSQP | None = None
        self._problem_shape: tuple[int, int, int] | None = None
        self._last_solution: Array | None = None
        self.last_profile_ms: dict[str, float] = {}

    def solve(
        self,
        robot: MuJoCoModelInterface,
        qpos_ref: Array,
        force_ref: Array | None = None,
        force_zero_weights: Array | None = None,
        stance_pos_refs: dict[str, Array] | None = None,
    ) -> StanceWBCSolution:
        total_start = perf_counter()
        profiler = _WBCSolveProfiler()
        cfg = self.config
        nv = robot.nv
        nu = robot.nu
        nf = 3 * len(cfg.foot_geoms)
        nvar = nv + nu + nf

        idx_vdot = slice(0, nv)
        idx_tau = slice(nv, nv + nu)
        idx_force = slice(nv + nu, nvar)

        with profiler.time("mass_matrix"):
            mass = robot.mass_matrix()
        with profiler.time("bias_forces"):
            h = robot.bias_forces()
        with profiler.time("actuation_matrix"):
            bmat = robot.actuation_matrix()
        with profiler.time("stance_jacobian"):
            jc = robot.stacked_geom_jacobian(list(cfg.foot_geoms))
        with profiler.time("jdot_v"):
            jdot_v = (
                robot.stacked_geom_jdot_v(list(cfg.foot_geoms))
                if cfg.use_jdot_v
                else np.zeros(3 * len(cfg.foot_geoms), dtype=float)
            )
        with profiler.time("task_commands"):
            stance_acc_cmd = self._stance_accel_cmd(robot, list(cfg.foot_geoms), stance_pos_refs)
            pos_acc_cmd = self._base_position_accel_cmd(robot, qpos_ref)
            ori_acc_cmd = self._base_orientation_accel_cmd(robot, qpos_ref)
            joint_acc_cmd = self._joint_accel_cmd(robot, qpos_ref)
        force_ref = self._force_reference(robot, len(cfg.foot_geoms)) if force_ref is None else np.asarray(force_ref, dtype=float)
        if force_ref.shape != (nf,):
            raise ValueError(f"force_ref must have shape ({nf},), got {force_ref.shape}")
        force_zero_weights = np.zeros(nf) if force_zero_weights is None else np.asarray(force_zero_weights, dtype=float)
        if force_zero_weights.shape != (nf,):
            raise ValueError(f"force_zero_weights must have shape ({nf},), got {force_zero_weights.shape}")

        with profiler.time("sparse_assembly"):
            p_diag = np.zeros(nvar)
            q = np.zeros(nvar)

            self._add_diagonal_tracking_cost(p_diag, q, slice(0, 3), cfg.weight_base_pos, pos_acc_cmd)
            self._add_diagonal_tracking_cost(p_diag, q, slice(3, 6), cfg.weight_base_ori, ori_acc_cmd)
            self._add_diagonal_tracking_cost(p_diag, q, slice(6, nv), cfg.weight_joint_posture, joint_acc_cmd)

            p_diag[idx_tau] += cfg.weight_tau
            p_diag[idx_force] += cfg.weight_force + force_zero_weights
            q[idx_force] += -cfg.weight_force * force_ref

            p = sparse.diags(p_diag + 1.0e-9, format="csc")

            aeq_dyn = sparse.hstack(
                [sparse.csc_matrix(mass), sparse.csc_matrix(-bmat), sparse.csc_matrix(-jc.T)],
                format="csc",
            )
            beq_dyn = -h

            aeq_stance = sparse.hstack(
                [
                    sparse.csc_matrix(jc),
                    sparse.csc_matrix((nf, nu)),
                    sparse.csc_matrix((nf, nf)),
                ],
                format="csc",
            )
            beq_stance = stance_acc_cmd - jdot_v

            a_friction, l_friction, u_friction = self._friction_constraints(nv, nu, nf, cfg.friction_mu, cfg.normal_force_min)
            a_tau, l_tau, u_tau = self._torque_constraints(robot, nv, nu, nf)

            a = sparse.vstack([aeq_dyn, aeq_stance, a_friction, a_tau], format="csc")
            l = np.concatenate([beq_dyn, beq_stance, l_friction, l_tau])
            u = np.concatenate([beq_dyn, beq_stance, u_friction, u_tau])

        result = self._solve_osqp(p, q, a, l, u, profiler)

        if result.x is None:
            z = np.zeros(nvar)
        else:
            z = result.x

        vdot = z[idx_vdot]
        tau = z[idx_tau]
        contact_forces = z[idx_force]
        dynamics_residual = mass @ vdot + h - bmat @ tau - jc.T @ contact_forces
        stance_residual = jc @ vdot + jdot_v - stance_acc_cmd
        self.last_profile_ms = profiler.milliseconds(total_start)

        return StanceWBCSolution(
            status=result.info.status,
            vdot=vdot,
            tau=tau,
            contact_forces=contact_forces,
            dynamics_residual=dynamics_residual,
            stance_residual=stance_residual,
            objective=float(result.info.obj_val),
        )

    def _solve_osqp(
        self,
        p: sparse.csc_matrix,
        q: Array,
        a: sparse.csc_matrix,
        l: Array,
        u: Array,
        profiler: _WBCSolveProfiler | None = None,
    ) -> Any:
        shape = (p.shape[0], a.shape[0], a.nnz)
        if self._solver is None or self._problem_shape != shape:
            self._solver = osqp.OSQP()
            with _maybe_profile(profiler, "osqp_setup"):
                self._solver.setup(
                    P=p,
                    q=q,
                    A=a,
                    l=l,
                    u=u,
                    verbose=False,
                    polish=True,
                    warm_starting=True,
                    eps_abs=1.0e-6,
                    eps_rel=1.0e-6,
                )
            self._problem_shape = shape
        else:
            try:
                with _maybe_profile(profiler, "osqp_update"):
                    self._solver.update(q=q, l=l, u=u, Px=p.data, Ax=a.data)
            except ValueError:
                self._solver = osqp.OSQP()
                with _maybe_profile(profiler, "osqp_setup"):
                    self._solver.setup(
                        P=p,
                        q=q,
                        A=a,
                        l=l,
                        u=u,
                        verbose=False,
                        polish=True,
                        warm_starting=True,
                        eps_abs=1.0e-6,
                        eps_rel=1.0e-6,
                    )
                self._problem_shape = shape

        if self._last_solution is not None and self._last_solution.shape == q.shape:
            self._solver.warm_start(x=self._last_solution)
        with _maybe_profile(profiler, "osqp_solve"):
            result = self._solver.solve()
        if result.x is not None:
            self._last_solution = result.x.copy()
        return result

    def _base_position_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        pos_error = qpos_ref[0:3] - robot.data.qpos[0:3]
        return cfg.kp_base_pos * pos_error - cfg.kd_base_pos * robot.data.qvel[0:3]

    def _base_orientation_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        current = robot.data.qpos[3:7]
        desired = qpos_ref[3:7]
        rotvec_error = quat_error_rotvec(desired, current)
        return cfg.kp_base_ori * rotvec_error - cfg.kd_base_ori * robot.data.qvel[3:6]

    def _joint_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        q_error = qpos_ref[7:] - robot.data.qpos[7:]
        return cfg.kp_joint * q_error - cfg.kd_joint * robot.data.qvel[6:]

    def _force_reference(self, robot: MuJoCoModelInterface, num_contacts: int) -> Array:
        total_mass = float(np.sum(robot.model.body_mass))
        gravity_z = abs(float(robot.model.opt.gravity[2]))
        f_ref = np.zeros(3 * num_contacts)
        f_ref[2::3] = total_mass * gravity_z / num_contacts
        return f_ref

    def _stance_accel_cmd(
        self,
        robot: MuJoCoModelInterface,
        foot_geoms: list[str],
        stance_pos_refs: dict[str, Array] | None,
    ) -> Array:
        cfg = self.config
        if stance_pos_refs is None or (cfg.kp_stance == 0.0 and cfg.kd_stance == 0.0):
            return np.zeros(3 * len(foot_geoms))

        commands = []
        for foot in foot_geoms:
            if foot not in stance_pos_refs:
                commands.append(np.zeros(3))
                continue
            pos_error = np.asarray(stance_pos_refs[foot], dtype=float) - robot.geom_position(foot)
            vel_error = -robot.geom_velocity(foot)
            commands.append(cfg.kp_stance * pos_error + cfg.kd_stance * vel_error)
        return np.concatenate(commands)

    @staticmethod
    def _add_diagonal_tracking_cost(p_diag: Array, q: Array, idx: slice, weight: float, target: Array) -> None:
        p_diag[idx] += weight
        q[idx] += -weight * target

    @staticmethod
    def _friction_constraints(
        nv: int,
        nu: int,
        nf: int,
        mu: float,
        normal_force_min: float = 0.0,
    ) -> tuple[sparse.csc_matrix, Array, Array]:
        rows = []
        cols = []
        vals = []
        upper = []
        nvar = nv + nu + nf

        def add(row: int, force_col: int, value: float) -> None:
            rows.append(row)
            cols.append(nv + nu + force_col)
            vals.append(value)

        row = 0
        for contact in range(nf // 3):
            fx = 3 * contact
            fy = fx + 1
            fz = fx + 2

            add(row, fx, 1.0)
            add(row, fz, -mu)
            upper.append(0.0)
            row += 1

            add(row, fx, -1.0)
            add(row, fz, -mu)
            upper.append(0.0)
            row += 1

            add(row, fy, 1.0)
            add(row, fz, -mu)
            upper.append(0.0)
            row += 1

            add(row, fy, -1.0)
            add(row, fz, -mu)
            upper.append(0.0)
            row += 1

            add(row, fz, -1.0)
            upper.append(-normal_force_min)
            row += 1

        amat = sparse.csc_matrix((vals, (rows, cols)), shape=(row, nvar))
        lower = np.full(row, -np.inf)
        return amat, lower, np.asarray(upper, dtype=float)

    @staticmethod
    def _torque_constraints(robot: MuJoCoModelInterface, nv: int, nu: int, nf: int) -> tuple[sparse.csc_matrix, Array, Array]:
        rows = np.arange(nu)
        cols = nv + np.arange(nu)
        vals = np.ones(nu)
        amat = sparse.csc_matrix((vals, (rows, cols)), shape=(nu, nv + nu + nf))
        ctrlrange = robot.model.actuator_ctrlrange.copy()
        return amat, ctrlrange[:, 0], ctrlrange[:, 1]


class SingleLegSwingWBCQP:
    """Three-foot stance plus one swing-foot acceleration task."""

    def __init__(self, config: SingleLegSwingWBCConfig | None = None):
        self.config = config or SingleLegSwingWBCConfig()
        self._solver: osqp.OSQP | None = None
        self._problem_shape: tuple[int, int, int, int] | None = None
        self._last_solution: Array | None = None
        self.last_profile_ms: dict[str, float] = {}

    def solve(
        self,
        robot: MuJoCoModelInterface,
        qpos_ref: Array,
        swing_pos_ref: Array,
        swing_vel_ref: Array | None = None,
        swing_acc_ref: Array | None = None,
        force_ref: Array | None = None,
        force_zero_weights: Array | None = None,
        stance_pos_refs: dict[str, Array] | None = None,
    ) -> SingleLegSwingWBCSolution:
        total_start = perf_counter()
        profiler = _WBCSolveProfiler()
        cfg = self.config
        nv = robot.nv
        nu = robot.nu
        nf = 3 * len(cfg.stance_foot_geoms)
        nvar = nv + nu + nf

        swing_vel_ref = np.zeros(3) if swing_vel_ref is None else np.asarray(swing_vel_ref, dtype=float)
        swing_acc_ref = np.zeros(3) if swing_acc_ref is None else np.asarray(swing_acc_ref, dtype=float)

        idx_vdot = slice(0, nv)
        idx_tau = slice(nv, nv + nu)
        idx_force = slice(nv + nu, nvar)

        with profiler.time("mass_matrix"):
            mass = robot.mass_matrix()
        with profiler.time("bias_forces"):
            h = robot.bias_forces()
        with profiler.time("actuation_matrix"):
            bmat = robot.actuation_matrix()
        with profiler.time("stance_jacobian"):
            jc = robot.stacked_geom_jacobian(list(cfg.stance_foot_geoms))
        with profiler.time("jdot_v"):
            jdot_v = (
                robot.stacked_geom_jdot_v(list(cfg.stance_foot_geoms))
                if cfg.use_jdot_v
                else np.zeros(3 * len(cfg.stance_foot_geoms), dtype=float)
            )
        with profiler.time("task_commands"):
            stance_acc_cmd = self._stance_accel_cmd(robot, list(cfg.stance_foot_geoms), stance_pos_refs)
            pos_acc_cmd = self._base_position_accel_cmd(robot, qpos_ref)
            ori_acc_cmd = self._base_orientation_accel_cmd(robot, qpos_ref)
            joint_acc_cmd = self._joint_accel_cmd(robot, qpos_ref)
        force_ref = (
            self._force_reference(robot, len(cfg.stance_foot_geoms))
            if force_ref is None
            else np.asarray(force_ref, dtype=float)
        )
        if force_ref.shape != (nf,):
            raise ValueError(f"force_ref must have shape ({nf},), got {force_ref.shape}")
        force_zero_weights = np.zeros(nf) if force_zero_weights is None else np.asarray(force_zero_weights, dtype=float)
        if force_zero_weights.shape != (nf,):
            raise ValueError(f"force_zero_weights must have shape ({nf},), got {force_zero_weights.shape}")

        with profiler.time("swing_jacobian"):
            swing_j = robot.geom_jacobian(cfg.swing_foot_geom).jacp
        with profiler.time("jdot_v"):
            swing_jdot_v = robot.geom_jdot_v(cfg.swing_foot_geom) if cfg.use_jdot_v else np.zeros(3, dtype=float)
        with profiler.time("task_commands"):
            swing_pos = robot.geom_position(cfg.swing_foot_geom)
            swing_vel = swing_j @ robot.data.qvel
            swing_acc_cmd = (
                swing_acc_ref
                + cfg.kp_swing * (np.asarray(swing_pos_ref, dtype=float) - swing_pos)
                + cfg.kd_swing * (swing_vel_ref - swing_vel)
            )
            swing_target = swing_acc_cmd - swing_jdot_v

        with profiler.time("sparse_assembly"):
            p = sparse.lil_matrix((nvar, nvar), dtype=float)
            q = np.zeros(nvar)

            self._add_diagonal_tracking_cost(p, q, slice(0, 3), cfg.weight_base_pos, pos_acc_cmd)
            self._add_diagonal_tracking_cost(p, q, slice(3, 6), cfg.weight_base_ori, ori_acc_cmd)
            self._add_diagonal_tracking_cost(p, q, slice(6, nv), cfg.weight_joint_posture, joint_acc_cmd)

            p[idx_tau, idx_tau] = p[idx_tau, idx_tau] + sparse.eye(nu, format="lil") * cfg.weight_tau
            p[idx_force, idx_force] = p[idx_force, idx_force] + sparse.diags(
                cfg.weight_force + force_zero_weights,
                format="lil",
            )
            q[idx_force] += -cfg.weight_force * force_ref

            swing_hessian = cfg.weight_swing_foot * (swing_j.T @ swing_j)
            p[idx_vdot, idx_vdot] = p[idx_vdot, idx_vdot] + sparse.csc_matrix(swing_hessian)
            q[idx_vdot] += -cfg.weight_swing_foot * swing_j.T @ swing_target

            p = sparse.triu(p + sparse.eye(nvar, format="lil") * 1.0e-9, format="csc")

            aeq_dyn = sparse.hstack(
                [sparse.csc_matrix(mass), sparse.csc_matrix(-bmat), sparse.csc_matrix(-jc.T)],
                format="csc",
            )
            beq_dyn = -h

            aeq_stance = sparse.hstack(
                [
                    sparse.csc_matrix(jc),
                    sparse.csc_matrix((nf, nu)),
                    sparse.csc_matrix((nf, nf)),
                ],
                format="csc",
            )
            beq_stance = stance_acc_cmd - jdot_v

            a_friction, l_friction, u_friction = StanceWBCQP._friction_constraints(
                nv,
                nu,
                nf,
                cfg.friction_mu,
                cfg.normal_force_min,
            )
            a_tau, l_tau, u_tau = StanceWBCQP._torque_constraints(robot, nv, nu, nf)

            a = sparse.vstack([aeq_dyn, aeq_stance, a_friction, a_tau], format="csc")
            l = np.concatenate([beq_dyn, beq_stance, l_friction, l_tau])
            u = np.concatenate([beq_dyn, beq_stance, u_friction, u_tau])

        result = self._solve_osqp(p, q, a, l, u, profiler)

        z = np.zeros(nvar) if result.x is None else result.x
        vdot = z[idx_vdot]
        tau = z[idx_tau]
        contact_forces = z[idx_force]
        dynamics_residual = mass @ vdot + h - bmat @ tau - jc.T @ contact_forces
        stance_residual = jc @ vdot + jdot_v - stance_acc_cmd
        swing_accel_error = swing_j @ vdot + swing_jdot_v - swing_acc_cmd
        self.last_profile_ms = profiler.milliseconds(total_start)

        return SingleLegSwingWBCSolution(
            status=result.info.status,
            vdot=vdot,
            tau=tau,
            contact_forces=contact_forces,
            dynamics_residual=dynamics_residual,
            stance_residual=stance_residual,
            objective=float(result.info.obj_val),
            swing_accel_error=swing_accel_error,
        )

    def _solve_osqp(
        self,
        p: sparse.csc_matrix,
        q: Array,
        a: sparse.csc_matrix,
        l: Array,
        u: Array,
        profiler: _WBCSolveProfiler | None = None,
    ) -> Any:
        shape = (p.shape[0], a.shape[0], p.nnz, a.nnz)
        if self._solver is None or self._problem_shape != shape:
            self._solver = osqp.OSQP()
            with _maybe_profile(profiler, "osqp_setup"):
                self._solver.setup(
                    P=p,
                    q=q,
                    A=a,
                    l=l,
                    u=u,
                    verbose=False,
                    polish=True,
                    warm_starting=True,
                    eps_abs=1.0e-6,
                    eps_rel=1.0e-6,
                )
            self._problem_shape = shape
        else:
            try:
                with _maybe_profile(profiler, "osqp_update"):
                    self._solver.update(q=q, l=l, u=u, Px=p.data, Ax=a.data)
            except ValueError:
                self._solver = osqp.OSQP()
                with _maybe_profile(profiler, "osqp_setup"):
                    self._solver.setup(
                        P=p,
                        q=q,
                        A=a,
                        l=l,
                        u=u,
                        verbose=False,
                        polish=True,
                        warm_starting=True,
                        eps_abs=1.0e-6,
                        eps_rel=1.0e-6,
                    )
                self._problem_shape = shape

        if self._last_solution is not None and self._last_solution.shape == q.shape:
            self._solver.warm_start(x=self._last_solution)
        with _maybe_profile(profiler, "osqp_solve"):
            result = self._solver.solve()
        if result.x is not None:
            self._last_solution = result.x.copy()
        return result

    def _base_position_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        pos_error = qpos_ref[0:3] - robot.data.qpos[0:3]
        return cfg.kp_base_pos * pos_error - cfg.kd_base_pos * robot.data.qvel[0:3]

    def _base_orientation_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        rotvec_error = quat_error_rotvec(qpos_ref[3:7], robot.data.qpos[3:7])
        return cfg.kp_base_ori * rotvec_error - cfg.kd_base_ori * robot.data.qvel[3:6]

    def _joint_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        q_error = qpos_ref[7:] - robot.data.qpos[7:]
        return cfg.kp_joint * q_error - cfg.kd_joint * robot.data.qvel[6:]

    def _force_reference(self, robot: MuJoCoModelInterface, num_contacts: int) -> Array:
        total_mass = float(np.sum(robot.model.body_mass))
        gravity_z = abs(float(robot.model.opt.gravity[2]))
        f_ref = np.zeros(3 * num_contacts)
        f_ref[2::3] = total_mass * gravity_z / num_contacts
        return f_ref

    def _stance_accel_cmd(
        self,
        robot: MuJoCoModelInterface,
        foot_geoms: list[str],
        stance_pos_refs: dict[str, Array] | None,
    ) -> Array:
        cfg = self.config
        if stance_pos_refs is None or (cfg.kp_stance == 0.0 and cfg.kd_stance == 0.0):
            return np.zeros(3 * len(foot_geoms))

        commands = []
        for foot in foot_geoms:
            if foot not in stance_pos_refs:
                commands.append(np.zeros(3))
                continue
            pos_error = np.asarray(stance_pos_refs[foot], dtype=float) - robot.geom_position(foot)
            vel_error = -robot.geom_velocity(foot)
            commands.append(cfg.kp_stance * pos_error + cfg.kd_stance * vel_error)
        return np.concatenate(commands)

    @staticmethod
    def _add_diagonal_tracking_cost(p: sparse.lil_matrix, q: Array, idx: slice, weight: float, target: Array) -> None:
        length = len(range(idx.start or 0, idx.stop or 0))
        p[idx, idx] = p[idx, idx] + sparse.eye(length, format="lil") * weight
        q[idx] += -weight * target


class GeneralContactWBCQP:
    """Full-body WBC for arbitrary non-flight contact modes.

    中文说明：
        这是当前 locomotion 主链路使用的通用 WBC。stance feet 作为硬接触
        约束进入 QP，swing feet 作为软足端加速度任务进入 cost。QP 在完整
        floating-base 动力学约束下求解 [vdot, tau, f]。

    This is the contact-mode generic version of the current WBC:

        stance_feet -> hard contact acceleration constraints and contact forces
        swing_feet  -> soft task-space acceleration tracking costs

    It supports crawl-like modes (3 stance + 1 swing) and trot-like modes
    (2 stance + 2 swing). Flight phases are intentionally out of scope because
    there are no contact forces and the floating base becomes underactuated.
    """

    def __init__(self, config: GeneralContactWBCConfig | None = None):
        self.config = config or GeneralContactWBCConfig()
        self._solver: osqp.OSQP | None = None
        self._problem_shape: tuple[int, int, int, int] | None = None
        self._last_solution: Array | None = None
        self.last_profile_ms: dict[str, float] = {}

    def solve(
        self,
        robot: MuJoCoModelInterface,
        qpos_ref: Array,
        swing_pos_refs: dict[str, Array] | None = None,
        swing_vel_refs: dict[str, Array] | None = None,
        swing_acc_refs: dict[str, Array] | None = None,
        force_ref: Array | None = None,
        force_zero_weights: Array | None = None,
        stance_pos_refs: dict[str, Array] | None = None,
    ) -> GeneralContactWBCSolution:
        total_start = perf_counter()
        profiler = _WBCSolveProfiler()
        cfg = self.config
        if len(cfg.stance_foot_geoms) == 0:
            raise ValueError("GeneralContactWBCQP requires at least one stance foot; flight needs a separate WBC.")

        swing_pos_refs = {} if swing_pos_refs is None else swing_pos_refs
        swing_vel_refs = {} if swing_vel_refs is None else swing_vel_refs
        swing_acc_refs = {} if swing_acc_refs is None else swing_acc_refs

        nv = robot.nv
        nu = robot.nu
        nf = 3 * len(cfg.stance_foot_geoms)
        nvar = nv + nu + nf

        idx_vdot = slice(0, nv)
        idx_tau = slice(nv, nv + nu)
        idx_force = slice(nv + nu, nvar)

        with profiler.time("mass_matrix"):
            mass = robot.mass_matrix()
        with profiler.time("bias_forces"):
            h = robot.bias_forces()
        with profiler.time("actuation_matrix"):
            bmat = robot.actuation_matrix()
        with profiler.time("stance_jacobian"):
            jc = robot.stacked_geom_jacobian(list(cfg.stance_foot_geoms))
        with profiler.time("jdot_v"):
            jdot_v = (
                robot.stacked_geom_jdot_v(list(cfg.stance_foot_geoms))
                if cfg.use_jdot_v
                else np.zeros(3 * len(cfg.stance_foot_geoms), dtype=float)
            )
        with profiler.time("task_commands"):
            stance_acc_cmd = self._stance_accel_cmd(robot, list(cfg.stance_foot_geoms), stance_pos_refs)
            pos_acc_cmd = self._base_position_accel_cmd(robot, qpos_ref)
            ori_acc_cmd = self._base_orientation_accel_cmd(robot, qpos_ref)
            joint_acc_cmd = self._joint_accel_cmd(robot, qpos_ref)
        force_ref = (
            self._force_reference(robot, len(cfg.stance_foot_geoms))
            if force_ref is None
            else np.asarray(force_ref, dtype=float)
        )
        if force_ref.shape != (nf,):
            raise ValueError(f"force_ref must have shape ({nf},), got {force_ref.shape}")
        force_zero_weights = np.zeros(nf) if force_zero_weights is None else np.asarray(force_zero_weights, dtype=float)
        if force_zero_weights.shape != (nf,):
            raise ValueError(f"force_zero_weights must have shape ({nf},), got {force_zero_weights.shape}")

        swing_tasks = []
        swing_errors = []
        for foot in cfg.swing_foot_geoms:
            with profiler.time("swing_jacobian"):
                swing_j = robot.geom_jacobian(foot).jacp
            with profiler.time("jdot_v"):
                swing_jdot_v = robot.geom_jdot_v(foot) if cfg.use_jdot_v else np.zeros(3, dtype=float)
            with profiler.time("task_commands"):
                swing_pos_default = robot.geom_position(foot)
                swing_pos_ref = np.asarray(swing_pos_refs.get(foot, swing_pos_default), dtype=float)
                swing_vel_ref = np.asarray(swing_vel_refs.get(foot, np.zeros(3)), dtype=float)
                swing_acc_ref = np.asarray(swing_acc_refs.get(foot, np.zeros(3)), dtype=float)
                swing_vel = swing_j @ robot.data.qvel
                swing_acc_cmd = (
                    swing_acc_ref
                    + cfg.kp_swing * (swing_pos_ref - swing_pos_default)
                    + cfg.kd_swing * (swing_vel_ref - swing_vel)
                )
                swing_target = swing_acc_cmd - swing_jdot_v
            swing_tasks.append((swing_j, swing_target))
            swing_errors.append((foot, swing_j, swing_jdot_v, swing_acc_cmd))

        with profiler.time("sparse_assembly"):
            p = sparse.lil_matrix((nvar, nvar), dtype=float)
            q = np.zeros(nvar)

            self._add_diagonal_tracking_cost(p, q, slice(0, 3), cfg.weight_base_pos, pos_acc_cmd)
            self._add_diagonal_tracking_cost(p, q, slice(3, 6), cfg.weight_base_ori, ori_acc_cmd)
            self._add_diagonal_tracking_cost(p, q, slice(6, nv), cfg.weight_joint_posture, joint_acc_cmd)

            p[idx_tau, idx_tau] = p[idx_tau, idx_tau] + sparse.eye(nu, format="lil") * cfg.weight_tau
            p[idx_force, idx_force] = p[idx_force, idx_force] + sparse.diags(
                cfg.weight_force + force_zero_weights,
                format="lil",
            )
            q[idx_force] += -cfg.weight_force * force_ref

            for swing_j, swing_target in swing_tasks:
                p[idx_vdot, idx_vdot] = p[idx_vdot, idx_vdot] + sparse.csc_matrix(
                    cfg.weight_swing_foot * (swing_j.T @ swing_j)
                )
                q[idx_vdot] += -cfg.weight_swing_foot * swing_j.T @ swing_target

            p = sparse.triu(p + sparse.eye(nvar, format="lil") * 1.0e-9, format="csc")

            aeq_dyn = sparse.hstack(
                [sparse.csc_matrix(mass), sparse.csc_matrix(-bmat), sparse.csc_matrix(-jc.T)],
                format="csc",
            )
            beq_dyn = -h

            aeq_stance = sparse.hstack(
                [
                    sparse.csc_matrix(jc),
                    sparse.csc_matrix((nf, nu)),
                    sparse.csc_matrix((nf, nf)),
                ],
                format="csc",
            )
            beq_stance = stance_acc_cmd - jdot_v

            a_friction, l_friction, u_friction = StanceWBCQP._friction_constraints(
                nv,
                nu,
                nf,
                cfg.friction_mu,
                cfg.normal_force_min,
            )
            a_tau, l_tau, u_tau = StanceWBCQP._torque_constraints(robot, nv, nu, nf)

            a = sparse.vstack([aeq_dyn, aeq_stance, a_friction, a_tau], format="csc")
            l = np.concatenate([beq_dyn, beq_stance, l_friction, l_tau])
            u = np.concatenate([beq_dyn, beq_stance, u_friction, u_tau])

        result = self._solve_osqp(p, q, a, l, u, profiler)

        z = np.zeros(nvar) if result.x is None else result.x
        vdot = z[idx_vdot]
        tau = z[idx_tau]
        contact_forces = z[idx_force]
        dynamics_residual = mass @ vdot + h - bmat @ tau - jc.T @ contact_forces
        stance_residual = jc @ vdot + jdot_v - stance_acc_cmd
        swing_accel_error = (
            np.concatenate([swing_j @ vdot + swing_jdot_v - swing_acc_cmd for _, swing_j, swing_jdot_v, swing_acc_cmd in swing_errors])
            if swing_errors
            else np.zeros(0)
        )
        self.last_profile_ms = profiler.milliseconds(total_start)

        return GeneralContactWBCSolution(
            status=result.info.status,
            vdot=vdot,
            tau=tau,
            contact_forces=contact_forces,
            dynamics_residual=dynamics_residual,
            stance_residual=stance_residual,
            objective=float(result.info.obj_val),
            swing_accel_error=swing_accel_error,
            stance_foot_geoms=cfg.stance_foot_geoms,
            swing_foot_geoms=cfg.swing_foot_geoms,
        )

    def _solve_osqp(
        self,
        p: sparse.csc_matrix,
        q: Array,
        a: sparse.csc_matrix,
        l: Array,
        u: Array,
        profiler: _WBCSolveProfiler | None = None,
    ) -> Any:
        shape = (p.shape[0], a.shape[0], p.nnz, a.nnz)
        if self._solver is None or self._problem_shape != shape:
            self._solver = osqp.OSQP()
            with _maybe_profile(profiler, "osqp_setup"):
                self._solver.setup(
                    P=p,
                    q=q,
                    A=a,
                    l=l,
                    u=u,
                    verbose=False,
                    polish=True,
                    warm_starting=True,
                    eps_abs=1.0e-6,
                    eps_rel=1.0e-6,
                )
            self._problem_shape = shape
        else:
            try:
                with _maybe_profile(profiler, "osqp_update"):
                    self._solver.update(q=q, l=l, u=u, Px=p.data, Ax=a.data)
            except ValueError:
                self._solver = osqp.OSQP()
                with _maybe_profile(profiler, "osqp_setup"):
                    self._solver.setup(
                        P=p,
                        q=q,
                        A=a,
                        l=l,
                        u=u,
                        verbose=False,
                        polish=True,
                        warm_starting=True,
                        eps_abs=1.0e-6,
                        eps_rel=1.0e-6,
                    )
                self._problem_shape = shape

        if self._last_solution is not None and self._last_solution.shape == q.shape:
            self._solver.warm_start(x=self._last_solution)
        with _maybe_profile(profiler, "osqp_solve"):
            result = self._solver.solve()
        if result.x is not None:
            self._last_solution = result.x.copy()
        return result

    def _base_position_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        pos_error = qpos_ref[0:3] - robot.data.qpos[0:3]
        return cfg.kp_base_pos * pos_error - cfg.kd_base_pos * robot.data.qvel[0:3]

    def _base_orientation_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        rotvec_error = quat_error_rotvec(qpos_ref[3:7], robot.data.qpos[3:7])
        return cfg.kp_base_ori * rotvec_error - cfg.kd_base_ori * robot.data.qvel[3:6]

    def _joint_accel_cmd(self, robot: MuJoCoModelInterface, qpos_ref: Array) -> Array:
        cfg = self.config
        q_error = qpos_ref[7:] - robot.data.qpos[7:]
        return cfg.kp_joint * q_error - cfg.kd_joint * robot.data.qvel[6:]

    def _force_reference(self, robot: MuJoCoModelInterface, num_contacts: int) -> Array:
        total_mass = float(np.sum(robot.model.body_mass))
        gravity_z = abs(float(robot.model.opt.gravity[2]))
        f_ref = np.zeros(3 * num_contacts)
        f_ref[2::3] = total_mass * gravity_z / num_contacts
        return f_ref

    def _stance_accel_cmd(
        self,
        robot: MuJoCoModelInterface,
        foot_geoms: list[str],
        stance_pos_refs: dict[str, Array] | None,
    ) -> Array:
        cfg = self.config
        if stance_pos_refs is None or (cfg.kp_stance == 0.0 and cfg.kd_stance == 0.0):
            return np.zeros(3 * len(foot_geoms))

        commands = []
        for foot in foot_geoms:
            if foot not in stance_pos_refs:
                commands.append(np.zeros(3))
                continue
            pos_error = np.asarray(stance_pos_refs[foot], dtype=float) - robot.geom_position(foot)
            vel_error = -robot.geom_velocity(foot)
            commands.append(cfg.kp_stance * pos_error + cfg.kd_stance * vel_error)
        return np.concatenate(commands)

    @staticmethod
    def _add_diagonal_tracking_cost(p: sparse.lil_matrix, q: Array, idx: slice, weight: float, target: Array) -> None:
        length = len(range(idx.start or 0, idx.stop or 0))
        p[idx, idx] = p[idx, idx] + sparse.eye(length, format="lil") * weight
        q[idx] += -weight * target


def quat_error_rotvec(desired: Array, current: Array) -> Array:
    """Return small-angle rotation vector that moves current toward desired."""

    desired = normalize_quat(desired)
    current = normalize_quat(current)
    error = quat_mul(desired, quat_conj(current))
    if error[0] < 0.0:
        error = -error
    return 2.0 * error[1:4]


def normalize_quat(quat: Array) -> Array:
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero quaternion.")
    return np.asarray(quat, dtype=float) / norm


def quat_conj(quat: Array) -> Array:
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def quat_mul(left: Array, right: Array) -> Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )
