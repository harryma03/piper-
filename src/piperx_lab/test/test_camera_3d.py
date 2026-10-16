"""Known 3D points exercise both lens models and the Vidu depth encoding."""

import math

import cv2
import numpy as np
import pytest

from piperx_lab.handeye import deproject_pixel
from piperx_lab.camera_target import detect_colored_point


class Info:
    width = 640
    height = 480
    k = [341.0, 0.0, 325.4, 0.0, 341.0, 239.2, 0.0, 0.0, 1.0]
    d = [-0.019793, 0.015843, -0.111706, 0.264543, 0.0]
    distortion_model = 'kannala_brandt4'


def sdk_kb4_project(point, info):
    """Independent form of Vidu SDK example/deproject/main.cpp projection."""
    x, y, z = point
    x, y = x / z, y / z
    r = math.hypot(x, y)
    theta = math.atan(r)
    theta2 = theta * theta
    a, b, c, d = info.d[:4]
    scale = theta * (1 + theta2 * (a + theta2 * (b + theta2 * (c + theta2 * d)))) / r if r else 1.0
    return np.array([info.k[0] * x * scale + info.k[2],
                     info.k[4] * y * scale + info.k[5]])


@pytest.mark.parametrize('point', [(0.0, 0.0, 0.5), (0.15, -0.1, 0.6),
                                    (-0.2, 0.12, 0.7)])
def test_c1w_deprojects_sdk_kb4_pixels(point):
    pixel = sdk_kb4_project(point, Info())
    recovered = deproject_pixel(pixel, point[2], Info())
    assert np.allclose(recovered, point, atol=1e-6)


def test_brown_conrady_deprojects_projected_point():
    info = Info()
    info.distortion_model = 'brown_conrady'
    info.d = [0.05, -0.01, 0.001, -0.001, 0.0]
    point = np.array([0.12, -0.08, 0.55])
    projected, _ = cv2.projectPoints(point.reshape(1, 3), np.zeros(3), np.zeros(3),
                                     np.array(info.k).reshape(3, 3), np.array(info.d))
    assert np.allclose(deproject_pixel(projected[0, 0], point[2], info), point, atol=1e-6)


def test_red_region_returns_c1w_optical_xyz():
    info = Info()
    expected = np.array([0.12, -0.08, 0.55])
    u, v = np.rint(sdk_kb4_project(expected, info)).astype(int)
    color = np.zeros((info.height, info.width, 3), np.uint8)
    cv2.circle(color, (u, v), 9, (0, 0, 255), -1)
    depth = np.zeros((info.height, info.width), np.uint16)
    cv2.circle(depth, (u, v), 9, round(expected[2] * 65535 / 3.0), -1)
    target = detect_colored_point(color, depth, info, 3.0)
    assert target is not None
    assert np.allclose(target['xyz_m'], expected, atol=0.002)


def test_invalid_kb4_coefficients_rejected():
    info = Info()
    info.d = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match='four coefficients'):
        deproject_pixel((300, 240), 0.5, info)
