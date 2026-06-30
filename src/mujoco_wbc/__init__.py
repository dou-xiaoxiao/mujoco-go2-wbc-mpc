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
from .contact_transition import (
    landing_force_zero_weights,
    landing_ramped_force_ref,
    smoothstep01,
    update_touchdown_hysteresis,
)
from .model_interface import MuJoCoModelInterface
from .planning import (
    BodyReferencePlanner,
    CrawlCommand,
    CrawlGaitConfig,
    CrawlGaitPlanner,
    FootholdPlan,
    ReferenceBundle,
    RollingFootholdPlanner,
    SwingWindow,
)
from .profiling import LoopProfiler, TimingStats
from .reference_inputs import (
    LocomotionReferenceFrame,
    ModeSupportReport,
    classify_contact_mode,
    named_contact_patterns,
    validate_reference_frame,
)
from .support_polygon import SupportReference, body_reference_from_support, support_centroid
from .swing_trajectory import SwingReference, smoothstep, swing_foothold_reference
from .wbc_qp import (
    GeneralContactWBCConfig,
    GeneralContactWBCQP,
    GeneralContactWBCSolution,
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
    "BodyReferencePlanner",
    "FootholdPlan",
    "ReferenceBundle",
    "RollingFootholdPlanner",
    "LoopProfiler",
    "LocomotionReferenceFrame",
    "ModeSupportReport",
    "GeneralContactWBCConfig",
    "GeneralContactWBCQP",
    "GeneralContactWBCSolution",
    "SingleLegSwingWBCConfig",
    "SingleLegSwingWBCQP",
    "SingleLegSwingWBCSolution",
    "StanceWBCConfig",
    "StanceWBCQP",
    "StanceWBCSolution",
    "SwingReference",
    "SwingWindow",
    "TimingStats",
    "SupportReference",
    "body_reference_from_support",
    "classify_contact_mode",
    "named_contact_patterns",
    "scheduled_swing_contacts",
    "single_leg_swing_schedule",
    "smoothstep",
    "smoothstep01",
    "support_centroid",
    "landing_force_zero_weights",
    "landing_ramped_force_ref",
    "swing_foothold_reference",
    "update_touchdown_hysteresis",
    "validate_reference_frame",
]
