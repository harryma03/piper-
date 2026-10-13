"""Top-down grasp waypoint geometry expressed in the arm base frame.

This module is pure geometry: it never talks to ROS and never moves hardware.
It turns one observed point into the ordered Cartesian waypoints that a grasp
executor must reach, together with the tool orientation that makes the gripper
approach the table from directly above.
"""

import math

import numpy as np

# Rotating 180 degrees about the base X axis points the tool +Z axis straight
# down (-base Z), which is the usual two-finger top-down grasp pose.
TOOL_DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)


def tool_down_orientation() -> tuple:
    """Return the ``(x, y, z, w)`` quaternion for a straight-down tool axis."""
    return TOOL_DOWN_XYZW


def plan_top_down_grasp(target_base, approach_height: float = 0.10,
                        grasp_offset: float = 0.0) -> dict:
    """Build pre-grasp / grasp / retreat waypoints for a top-down grasp.

    ``approach_height`` is how far above the target the tool hovers before
    descending; ``grasp_offset`` lets the grasp point sit slightly above the
    reported surface (0.0 keeps it exactly at the observed point).
    """
    position = np.asarray(target_base, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError('target_base must be three finite numbers')
    if not math.isfinite(approach_height) or approach_height <= 0.0:
        raise ValueError('approach_height must be a positive number of metres')
    if not math.isfinite(grasp_offset) or grasp_offset < 0.0:
        raise ValueError('grasp_offset must be zero or a positive number of metres')
    if grasp_offset >= approach_height:
        raise ValueError('grasp_offset must be smaller than approach_height')

    orientation = tool_down_orientation()
    grasp = position + np.array([0.0, 0.0, grasp_offset])
    hover = grasp + np.array([0.0, 0.0, approach_height])
    return {
        'target_base': tuple(float(value) for value in position),
        'orientation_xyzw': orientation,
        'waypoints': [
            {'name': 'pre_grasp', 'position': tuple(float(v) for v in hover)},
            {'name': 'grasp', 'position': tuple(float(v) for v in grasp)},
            {'name': 'retreat', 'position': tuple(float(v) for v in hover)},
        ],
    }


def gripper_width_for_target(target_width: float, clearance: float = 0.01,
                             minimum: float = 0.0, maximum: float = 0.1) -> float:
    """Pick an opening that clears the target before closing on it."""
    if not math.isfinite(target_width) or target_width < 0.0:
        raise ValueError('target_width must be zero or a positive number of metres')
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError('clearance must be zero or a positive number of metres')
    opening = target_width + 2.0 * clearance
    return float(min(max(opening, minimum), maximum))
