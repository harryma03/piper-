import math

import numpy as np
import pytest

from piperx_lab.grasp_plan import (gripper_width_for_target, plan_top_down_grasp,
                                   tool_down_orientation)


def _quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_tool_down_quaternion_points_plus_z_down():
    matrix = _quaternion_to_matrix(*tool_down_orientation())
    assert np.allclose(matrix @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0])


def test_plan_orders_hover_grasp_retreat():
    plan = plan_top_down_grasp([0.1, 0.2, 0.05], approach_height=0.12)
    names = [waypoint['name'] for waypoint in plan['waypoints']]
    assert names == ['pre_grasp', 'grasp', 'retreat']
    pre_grasp, grasp, retreat = (waypoint['position'] for waypoint in plan['waypoints'])
    assert math.isclose(pre_grasp[2], 0.17, abs_tol=1e-9)
    assert math.isclose(grasp[2], 0.05, abs_tol=1e-9)
    assert np.allclose(pre_grasp[:2], [0.1, 0.2])
    assert np.allclose(retreat, pre_grasp)


def test_grasp_offset_raises_the_grasp_point():
    plan = plan_top_down_grasp([0.0, 0.0, 0.05], approach_height=0.1, grasp_offset=0.02)
    assert math.isclose(plan['waypoints'][1]['position'][2], 0.07, abs_tol=1e-9)


def test_plan_rejects_bad_geometry():
    with pytest.raises(ValueError):
        plan_top_down_grasp([0.0, 0.0, 0.0], approach_height=0.0)
    with pytest.raises(ValueError):
        plan_top_down_grasp([0.0, 0.0, 0.0], approach_height=0.05, grasp_offset=0.05)
    with pytest.raises(ValueError):
        plan_top_down_grasp([0.0, 0.0])


def test_gripper_width_clamps_and_clears_target():
    assert math.isclose(gripper_width_for_target(0.04, clearance=0.01), 0.06, abs_tol=1e-9)
    assert math.isclose(gripper_width_for_target(0.2, maximum=0.1), 0.1, abs_tol=1e-9)
