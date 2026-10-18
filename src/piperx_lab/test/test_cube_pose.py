"""Known-cube geometry should recover centre while rejecting partial faces."""

import math

import numpy as np
import pytest

from piperx_lab.cube_pose import estimate_upright_cube


def test_upright_cube_center_and_square_yaw():
    axis = np.linspace(-0.02, 0.02, 25)
    u, v = np.meshgrid(axis, axis)
    local = np.column_stack((u.ravel(), v.ravel()))
    yaw = 0.2
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    xy = local @ rotation.T + [0.31, -0.04]
    top = np.column_stack((xy, np.full(len(xy), 0.14)))
    side = top[::20].copy()
    side[:, 2] = 0.12
    result = estimate_upright_cube(np.vstack((top, side)), 0.04, table_height=0.10)
    np.testing.assert_allclose(result['center_xyz_m'], [0.31, -0.04, 0.12], atol=1e-5)
    assert abs(result['yaw_mod_90_rad'] - yaw) < 1e-3


def test_partial_top_does_not_claim_object_pose():
    u, v = np.meshgrid(np.linspace(0, 0.008, 12), np.linspace(0, 0.008, 12))
    points = np.column_stack((u.ravel(), v.ravel(), np.full(u.size, 0.04)))
    with pytest.raises(ValueError, match='footprint'):
        estimate_upright_cube(points, 0.04)
