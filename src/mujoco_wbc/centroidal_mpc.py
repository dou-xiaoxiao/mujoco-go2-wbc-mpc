"""Centroidal contact-force QP for the first MPC milestone.

This module intentionally starts with a single shooting node:

    decision variable = [f_FL, f_FR, f_RL, f_RR]

The QP chooses world-frame contact forces that match a desired centroidal
wrench at the robot COM while respecting friction pyramids. A later MPC layer
can stack this same wrench map over a horizon and add COM dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from .model_interface import MuJoCoModelInterface


Array = np.ndarray


@dataclass(frozen=True)
class CentroidalForceQPConfig:
    contact_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    friction_mu: float = 0.6
    normal_force_min: float = 0.0
    weight_wrench: float = 100.0
    weight_force_regularization: float = 1.0e-4


@dataclass(frozen=True)
class CentroidalForceQPSolution:
    status: str
    contact_forces: Array
    center_of_mass: Array
    contact_positions: Array
    desired_wrench: Array
    net_force: Array
    net_torque: Array
    wrench_error: Array
    objective: float


@dataclass(frozen=True)
class CentroidalMPCConfig:
    contact_geoms: tuple[str, ...] = ("FL", "FR", "RL", "RR")
    horizon_steps: int = 10
    dt: float = 0.03
    friction_mu: float = 0.6
    normal_force_min: float = 0.0
    weight_com_position: float = 500.0
    weight_com_velocity: float = 20.0
    weight_orientation: float = 200.0
    weight_angular_velocity: float = 20.0
    weight_force_regularization: float = 1.0e-4
    weight_force_rate: float = 1.0e-5


@dataclass(frozen=True)
class CentroidalMPCSolution:
    status: str
    states: Array
    contact_forces: Array
    first_contact_forces: Array
    contact_schedule: Array
    inertia_world: Array
    contact_positions: Array
    objective: float
    dynamics_residual: Array


class CentroidalForceQP:
    """Solve one centroidal contact-force allocation QP."""

    def __init__(self, config: CentroidalForceQPConfig | None = None):
        self.config = config or CentroidalForceQPConfig()

    def solve(
        self,
        robot: MuJoCoModelInterface,
        desired_linear_accel: Array | None = None,
        desired_torque: Array | None = None,
    ) -> CentroidalForceQPSolution:
        cfg = self.config
        num_contacts = len(cfg.contact_geoms)
        nvar = 3 * num_contacts

        desired_linear_accel = (
            np.zeros(3, dtype=float)
            if desired_linear_accel is None
            else np.asarray(desired_linear_accel, dtype=float)
        )
        desired_torque = (
            np.zeros(3, dtype=float)
            if desired_torque is None
            else np.asarray(desired_torque, dtype=float)
        )

        mass = robot.total_mass()
        gravity = np.asarray(robot.model.opt.gravity, dtype=float)
        com = robot.center_of_mass()
        contact_positions = np.vstack([robot.geom_position(name) for name in cfg.contact_geoms])
        wrench_map = self._wrench_map(contact_positions, com)
        desired_force = mass * (desired_linear_accel - gravity)
        desired_wrench = np.concatenate([desired_force, desired_torque])
        force_ref = self._force_reference(mass, gravity, num_contacts)

        p = (
            cfg.weight_wrench * (wrench_map.T @ wrench_map)
            + cfg.weight_force_regularization * np.eye(nvar)
            + 1.0e-9 * np.eye(nvar)
        )
        q = (
            -cfg.weight_wrench * wrench_map.T @ desired_wrench
            - cfg.weight_force_regularization * force_ref
        )

        a_friction, l_friction, u_friction = self._friction_constraints(
            num_contacts,
            cfg.friction_mu,
            cfg.normal_force_min,
        )

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(p),
            q=q,
            A=a_friction,
            l=l_friction,
            u=u_friction,
            verbose=False,
            polish=True,
            eps_abs=1.0e-7,
            eps_rel=1.0e-7,
        )
        result = solver.solve()

        contact_forces = np.zeros(nvar, dtype=float) if result.x is None else result.x
        achieved_wrench = wrench_map @ contact_forces

        return CentroidalForceQPSolution(
            status=result.info.status,
            contact_forces=contact_forces,
            center_of_mass=com,
            contact_positions=contact_positions,
            desired_wrench=desired_wrench,
            net_force=achieved_wrench[0:3],
            net_torque=achieved_wrench[3:6],
            wrench_error=achieved_wrench - desired_wrench,
            objective=float(result.info.obj_val),
        )

    @staticmethod
    def _wrench_map(contact_positions: Array, com: Array) -> Array:
        num_contacts = contact_positions.shape[0]
        wrench_map = np.zeros((6, 3 * num_contacts), dtype=float)

        for contact_id, contact_position in enumerate(contact_positions):
            block = slice(3 * contact_id, 3 * contact_id + 3)
            lever_arm = contact_position - com
            wrench_map[0:3, block] = np.eye(3)
            wrench_map[3:6, block] = skew(lever_arm)

        return wrench_map

    @staticmethod
    def _force_reference(mass: float, gravity: Array, num_contacts: int) -> Array:
        force_ref = np.zeros(3 * num_contacts, dtype=float)
        force_ref[2::3] = mass * abs(float(gravity[2])) / num_contacts
        return force_ref

    @staticmethod
    def _friction_constraints(
        num_contacts: int,
        mu: float,
        normal_force_min: float,
    ) -> tuple[sparse.csc_matrix, Array, Array]:
        rows = []
        cols = []
        vals = []
        upper = []

        def add(row: int, col: int, value: float) -> None:
            rows.append(row)
            cols.append(col)
            vals.append(value)

        row = 0
        for contact in range(num_contacts):
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

        amat = sparse.csc_matrix((vals, (rows, cols)), shape=(row, 3 * num_contacts))
        lower = np.full(row, -np.inf)
        return amat, lower, np.asarray(upper, dtype=float)


class CentroidalMPC:
    """Linear SRB MPC with per-foot contact forces and a fixed contact schedule.

    State:

        x = [com_position_W, com_velocity_W, theta_W, omega_W]

    Control:

        u = [f_FL, f_FR, f_RL, f_RR]

    Dynamics use the total robot COM:

        p[k+1] = p[k] + dt v[k]
        v[k+1] = v[k] + dt (sum_i f_i[k] / m + g)
        theta[k+1] = theta[k] + dt omega[k]
        omega[k+1] = omega[k] + dt I_W^-1 sum_i (p_i - com) x f_i[k]

    theta is a small-angle orientation error around the current MPC linearization
    pose, not a global quaternion state.
    """

    def __init__(self, config: CentroidalMPCConfig | None = None):
        self.config = config or CentroidalMPCConfig()

    def solve(
        self,
        robot: MuJoCoModelInterface,
        com_position_ref: Array,
        com_velocity_ref: Array | None = None,
        orientation_ref: Array | None = None,
        angular_velocity_ref: Array | None = None,
        contact_schedule: Array | None = None,
    ) -> CentroidalMPCSolution:
        cfg = self.config
        n_steps = cfg.horizon_steps
        n_contacts = len(cfg.contact_geoms)
        nx = 12
        nu = 3 * n_contacts
        n_state_vars = (n_steps + 1) * nx
        n_force_vars = n_steps * nu
        nvar = n_state_vars + n_force_vars

        com_position_ref = self._expand_reference(com_position_ref, n_steps + 1, 3, "com_position_ref")
        if com_velocity_ref is None:
            com_velocity_ref = np.zeros((n_steps + 1, 3), dtype=float)
        com_velocity_ref = self._expand_reference(com_velocity_ref, n_steps + 1, 3, "com_velocity_ref")
        if orientation_ref is None:
            orientation_ref = np.zeros((n_steps + 1, 3), dtype=float)
        orientation_ref = self._expand_reference(orientation_ref, n_steps + 1, 3, "orientation_ref")
        if angular_velocity_ref is None:
            angular_velocity_ref = np.zeros((n_steps + 1, 3), dtype=float)
        angular_velocity_ref = self._expand_reference(angular_velocity_ref, n_steps + 1, 3, "angular_velocity_ref")

        if contact_schedule is None:
            contact_schedule = np.ones((n_steps, n_contacts), dtype=bool)
        contact_schedule = np.asarray(contact_schedule, dtype=bool)
        if contact_schedule.shape != (n_steps, n_contacts):
            raise ValueError(f"contact_schedule must have shape ({n_steps}, {n_contacts}), got {contact_schedule.shape}")

        mass = robot.total_mass()
        gravity = np.asarray(robot.model.opt.gravity, dtype=float)
        com = robot.center_of_mass()
        contact_positions = np.vstack([robot.geom_position(name) for name in cfg.contact_geoms])
        inertia_world = robot.composite_inertia_world_about_com()
        inertia_inv_world = np.linalg.inv(inertia_world)
        x0 = np.concatenate(
            [
                com,
                robot.base_linear_velocity(),
                np.zeros(3, dtype=float),
                robot.base_angular_velocity(),
            ]
        )

        p_diag = np.zeros(nvar, dtype=float)
        q = np.zeros(nvar, dtype=float)
        self._add_state_tracking_cost(
            p_diag,
            q,
            n_steps,
            nx,
            com_position_ref,
            com_velocity_ref,
            orientation_ref,
            angular_velocity_ref,
        )
        self._add_force_cost(p_diag, q, n_state_vars, n_steps, n_contacts, mass, gravity, contact_schedule)
        p = sparse.diags(p_diag + 1.0e-9, format="lil")
        self._add_force_rate_cost(p, q, n_state_vars, n_steps, nu)
        p = p.tocsc()

        a_dyn, l_dyn, u_dyn = self._dynamics_constraints(
            n_steps,
            nx,
            nu,
            n_state_vars,
            mass,
            gravity,
            x0,
            contact_positions,
            com,
            inertia_inv_world,
        )
        a_force, l_force, u_force = self._force_constraints(
            n_state_vars,
            n_steps,
            n_contacts,
            nvar,
            cfg.friction_mu,
            cfg.normal_force_min,
            contact_schedule,
        )
        a = sparse.vstack([a_dyn, a_force], format="csc")
        l = np.concatenate([l_dyn, l_force])
        u = np.concatenate([u_dyn, u_force])

        solver = osqp.OSQP()
        solver.setup(
            P=p,
            q=q,
            A=a,
            l=l,
            u=u,
            verbose=False,
            polish=True,
            eps_abs=1.0e-7,
            eps_rel=1.0e-7,
            max_iter=10000,
        )
        result = solver.solve()

        z = np.zeros(nvar, dtype=float) if result.x is None else result.x
        states = z[:n_state_vars].reshape(n_steps + 1, nx)
        contact_forces = z[n_state_vars:].reshape(n_steps, n_contacts, 3)
        dynamics_residual = self._compute_dynamics_residual(
            states,
            contact_forces,
            mass,
            gravity,
            cfg.dt,
            contact_positions,
            com,
            inertia_inv_world,
        )

        return CentroidalMPCSolution(
            status=result.info.status,
            states=states,
            contact_forces=contact_forces,
            first_contact_forces=contact_forces[0].reshape(-1),
            contact_schedule=contact_schedule,
            inertia_world=inertia_world,
            contact_positions=contact_positions,
            objective=float(result.info.obj_val),
            dynamics_residual=dynamics_residual,
        )

    def _add_state_tracking_cost(
        self,
        p_diag: Array,
        q: Array,
        n_steps: int,
        nx: int,
        com_position_ref: Array,
        com_velocity_ref: Array,
        orientation_ref: Array,
        angular_velocity_ref: Array,
    ) -> None:
        cfg = self.config
        for step in range(1, n_steps + 1):
            base = step * nx
            p_diag[base : base + 3] += cfg.weight_com_position
            q[base : base + 3] += -cfg.weight_com_position * com_position_ref[step]
            p_diag[base + 3 : base + 6] += cfg.weight_com_velocity
            q[base + 3 : base + 6] += -cfg.weight_com_velocity * com_velocity_ref[step]
            p_diag[base + 6 : base + 9] += cfg.weight_orientation
            q[base + 6 : base + 9] += -cfg.weight_orientation * orientation_ref[step]
            p_diag[base + 9 : base + 12] += cfg.weight_angular_velocity
            q[base + 9 : base + 12] += -cfg.weight_angular_velocity * angular_velocity_ref[step]

    def _add_force_cost(
        self,
        p_diag: Array,
        q: Array,
        force_offset: int,
        n_steps: int,
        n_contacts: int,
        mass: float,
        gravity: Array,
        contact_schedule: Array,
    ) -> None:
        cfg = self.config
        for step in range(n_steps):
            stance_count = max(int(np.count_nonzero(contact_schedule[step])), 1)
            fz_ref = mass * abs(float(gravity[2])) / stance_count
            for contact in range(n_contacts):
                block = force_offset + step * 3 * n_contacts + 3 * contact
                p_diag[block : block + 3] += cfg.weight_force_regularization
                if contact_schedule[step, contact]:
                    q[block + 2] += -cfg.weight_force_regularization * fz_ref

    def _add_force_rate_cost(self, p: sparse.lil_matrix, q: Array, force_offset: int, n_steps: int, nu: int) -> None:
        del q
        weight = self.config.weight_force_rate
        if weight <= 0.0:
            return
        for step in range(1, n_steps):
            prev = force_offset + (step - 1) * nu
            curr = force_offset + step * nu
            for idx in range(nu):
                p[prev + idx, prev + idx] += weight
                p[curr + idx, curr + idx] += weight
                p[prev + idx, curr + idx] += -weight
                p[curr + idx, prev + idx] += -weight

    def _dynamics_constraints(
        self,
        n_steps: int,
        nx: int,
        nu: int,
        force_offset: int,
        mass: float,
        gravity: Array,
        x0: Array,
        contact_positions: Array,
        com: Array,
        inertia_inv_world: Array,
    ) -> tuple[sparse.csc_matrix, Array, Array]:
        cfg = self.config
        rows = []
        cols = []
        vals = []
        bounds = []
        row = 0

        def add(col: int, value: float) -> None:
            rows.append(row)
            cols.append(col)
            vals.append(value)

        for idx in range(nx):
            add(idx, 1.0)
            bounds.append(x0[idx])
            row += 1

        for step in range(n_steps):
            xk = step * nx
            xkp1 = (step + 1) * nx
            uk = force_offset + step * nu

            for axis in range(3):
                add(xkp1 + axis, 1.0)
                add(xk + axis, -1.0)
                add(xk + 3 + axis, -cfg.dt)
                bounds.append(0.0)
                row += 1

            for axis in range(3):
                add(xkp1 + 3 + axis, 1.0)
                add(xk + 3 + axis, -1.0)
                for force_axis in range(axis, nu, 3):
                    add(uk + force_axis, -cfg.dt / mass)
                bounds.append(cfg.dt * gravity[axis])
                row += 1

            for axis in range(3):
                add(xkp1 + 6 + axis, 1.0)
                add(xk + 6 + axis, -1.0)
                add(xk + 9 + axis, -cfg.dt)
                bounds.append(0.0)
                row += 1

            torque_map = self._angular_accel_map(contact_positions, com, inertia_inv_world)
            for axis in range(3):
                add(xkp1 + 9 + axis, 1.0)
                add(xk + 9 + axis, -1.0)
                for force_idx in range(nu):
                    add(uk + force_idx, -cfg.dt * torque_map[axis, force_idx])
                bounds.append(0.0)
                row += 1

        amat = sparse.csc_matrix((vals, (rows, cols)), shape=(row, force_offset + n_steps * nu))
        return amat, np.asarray(bounds, dtype=float), np.asarray(bounds, dtype=float)

    @staticmethod
    def _force_constraints(
        force_offset: int,
        n_steps: int,
        n_contacts: int,
        nvar: int,
        mu: float,
        normal_force_min: float,
        contact_schedule: Array,
    ) -> tuple[sparse.csc_matrix, Array, Array]:
        rows = []
        cols = []
        vals = []
        lower = []
        upper = []
        row = 0

        def add(col: int, value: float) -> None:
            rows.append(row)
            cols.append(col)
            vals.append(value)

        for step in range(n_steps):
            for contact in range(n_contacts):
                base = force_offset + step * 3 * n_contacts + 3 * contact
                fx = base
                fy = base + 1
                fz = base + 2

                if not contact_schedule[step, contact]:
                    for axis_col in (fx, fy, fz):
                        add(axis_col, 1.0)
                        lower.append(0.0)
                        upper.append(0.0)
                        row += 1
                    continue

                add(fx, 1.0)
                add(fz, -mu)
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

                add(fx, -1.0)
                add(fz, -mu)
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

                add(fy, 1.0)
                add(fz, -mu)
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

                add(fy, -1.0)
                add(fz, -mu)
                lower.append(-np.inf)
                upper.append(0.0)
                row += 1

                add(fz, 1.0)
                lower.append(normal_force_min)
                upper.append(np.inf)
                row += 1

        amat = sparse.csc_matrix((vals, (rows, cols)), shape=(row, nvar))
        return amat, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)

    @staticmethod
    def _compute_dynamics_residual(
        states: Array,
        forces: Array,
        mass: float,
        gravity: Array,
        dt: float,
        contact_positions: Array,
        com: Array,
        inertia_inv_world: Array,
    ) -> Array:
        residuals = []
        angular_accel_map = CentroidalMPC._angular_accel_map(contact_positions, com, inertia_inv_world)
        for step in range(forces.shape[0]):
            pos_next = states[step, 0:3] + dt * states[step, 3:6]
            accel = np.sum(forces[step], axis=0) / mass + gravity
            vel_next = states[step, 3:6] + dt * accel
            theta_next = states[step, 6:9] + dt * states[step, 9:12]
            omega_next = states[step, 9:12] + dt * (angular_accel_map @ forces[step].reshape(-1))
            residuals.append(states[step + 1, 0:3] - pos_next)
            residuals.append(states[step + 1, 3:6] - vel_next)
            residuals.append(states[step + 1, 6:9] - theta_next)
            residuals.append(states[step + 1, 9:12] - omega_next)
        return np.concatenate(residuals)

    @staticmethod
    def _angular_accel_map(contact_positions: Array, com: Array, inertia_inv_world: Array) -> Array:
        n_contacts = contact_positions.shape[0]
        torque_map = np.zeros((3, 3 * n_contacts), dtype=float)
        for contact in range(n_contacts):
            block = slice(3 * contact, 3 * contact + 3)
            torque_map[:, block] = skew(contact_positions[contact] - com)
        return inertia_inv_world @ torque_map

    @staticmethod
    def _expand_reference(reference: Array, rows: int, cols: int, name: str) -> Array:
        reference = np.asarray(reference, dtype=float)
        if reference.shape == (cols,):
            return np.tile(reference, (rows, 1))
        if reference.shape == (rows, cols):
            return reference
        raise ValueError(f"{name} must have shape ({cols},) or ({rows}, {cols}), got {reference.shape}")


def skew(vector: Array) -> Array:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )
