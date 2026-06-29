"""Support polygon helpers for slow gait/body-reference scheduling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SupportReference:
    stance_feet: tuple[str, ...]
    support_points_xy: Array
    centroid_xy: Array
    body_xy_ref: Array


def support_centroid(points_xy: Array) -> Array:
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("support points must have shape (n, 2)")
    return np.mean(points, axis=0)


def shrink_toward_centroid(point_xy: Array, centroid_xy: Array, ratio: float) -> Array:
    """Move a point toward the support centroid.

    ratio=1 returns the centroid, ratio=0 returns the original point.
    """

    point = np.asarray(point_xy, dtype=float)
    centroid = np.asarray(centroid_xy, dtype=float)
    return point + float(ratio) * (centroid - point)


def body_reference_from_support(
    current_body_xy: Array,
    support_points_xy: Array,
    centroid_ratio: float = 0.85,
) -> Array:
    """Choose a conservative body xy reference inside the support polygon.

    The first version simply moves the current body xy toward the stance-foot
    centroid. It is intentionally conservative and quasi-static.
    """

    centroid = support_centroid(support_points_xy)
    return shrink_toward_centroid(current_body_xy, centroid, centroid_ratio)
