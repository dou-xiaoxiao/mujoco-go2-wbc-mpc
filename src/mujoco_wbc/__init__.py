"""MuJoCo WBC/MPC utilities for floating-base quadrupeds."""

from . import conventions
from .centroidal_mpc import (
    CentroidalForceQP,
    CentroidalForceQPConfig,
    CentroidalForceQPSolution,
    CentroidalMPC,
    CentroidalMPCConfig,
    CentroidalMPCSolution,
)
from .contact_schedule import (
    active_swing_window,
    current_single_leg_phase,
    scheduled_swing_contacts,
    single_leg_swing_schedule,
)
from .model_interface import MuJoCoModelInterface
from .planning import CrawlCommand, CrawlGaitConfig, CrawlGaitPlanner, SwingWindow
from .support_polygon import SupportReference, body_reference_from_support, support_centroid
from .swing_trajectory import SwingReference, smoothstep, swing_foothold_reference
from .wbc_qp import (
    SingleLegSwingWBCConfig,
    SingleLegSwingWBCQP,
    SingleLegSwingWBCSolution,
    StanceWBCConfig,
    StanceWBCQP,
    StanceWBCSolution,
)

__all__ = [
    "MuJoCoModelInterface",
    "conventions",
    "CentroidalForceQP",
    "CentroidalForceQPConfig",
    "CentroidalForceQPSolution",
    "CentroidalMPC",
    "CentroidalMPCConfig",
    "CentroidalMPCSolution",
    "active_swing_window",
    "current_single_leg_phase",
    "CrawlGaitConfig",
    "CrawlCommand",
    "CrawlGaitPlanner",
    "SingleLegSwingWBCConfig",
    "SingleLegSwingWBCQP",
    "SingleLegSwingWBCSolution",
    "StanceWBCConfig",
    "StanceWBCQP",
    "StanceWBCSolution",
    "SwingReference",
    "SwingWindow",
    "SupportReference",
    "body_reference_from_support",
    "scheduled_swing_contacts",
    "single_leg_swing_schedule",
    "smoothstep",
    "support_centroid",
    "swing_foothold_reference",
]
