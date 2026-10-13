import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from piperx_lab.handeye import board_extent_fraction, board_pose, framing_verdict, load_board

BOARD_JSON = Path(__file__).resolve().parents[3] / 'calibration' / 'board.json'


class FakeInfo:
    distortion_model = 'brown_conrady'
    width = 640
    height = 480
    k = [542.0, 0.0, 320.0, 0.0, 542.0, 240.0, 0.0, 0.0, 1.0]
    d = [0.0, 0.0, 0.0, 0.0, 0.0]


def _transform(z):
    t = np.eye(4)
    t[:3, 3] = [0.0, 0.0, z]
    return t


def test_extent_shrinks_with_distance():
    near, _ = board_extent_fraction(_transform(0.4), FakeInfo(), 0.2)
    far, _ = board_extent_fraction(_transform(0.8), FakeInfo(), 0.2)
    assert near > far
    # 0.2 m board at 0.4 m with fx=542 spans ~0.68 of a 640 px frame
    assert math.isclose(near, 0.2 * 542 / 0.4 / 640, rel_tol=0.05)


def test_good_view_passes():
    ok, problems = framing_verdict(0.35, 20, 0.5, 0.5)
    assert ok and problems == []


def test_too_far_is_reported():
    ok, problems = framing_verdict(0.08, 20, 0.5, 1.2)
    assert not ok
    assert any('closer' in item for item in problems)


def test_few_markers_is_reported():
    ok, problems = framing_verdict(0.35, 3, 0.5, 0.5)
    assert not ok
    assert any('markers' in item for item in problems)


def test_blurry_board_is_reported():
    ok, problems = framing_verdict(0.35, 20, 4.5, 0.5)
    assert not ok
    assert any('RMS' in item for item in problems)


def render_board_view(transform, info):
    """Render the real board texture as it would appear to the camera."""
    image = np.full((info.height, info.width, 3), 255, np.uint8)
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64)
    dictionary_name, markers = load_board(BOARD_JSON)
    dictionary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dictionary_name))
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    for marker_id, corners in markers.items():
        projected, _ = cv2.projectPoints(corners, rvec, transform[:3, 3], k, d)
        destination = projected.reshape(4, 2).astype(np.float32)
        marker = cv2.aruco.drawMarker(dictionary, marker_id, 100)
        source = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32)
        homography = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(marker, homography, (info.width, info.height),
                                     flags=cv2.INTER_NEAREST, borderValue=255)
        image[warped < 128] = 0
    return image


def test_board_pose_recovers_a_synthetic_view():
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 0.5]
    bgr = render_board_view(transform, FakeInfo())
    recovered, error, marker_ids, _ = board_pose(bgr, FakeInfo(), BOARD_JSON)
    assert len(marker_ids) >= 9
    assert error < 2.0
    assert np.allclose(recovered[:3, 3], [0.0, 0.0, 0.5], atol=0.01)
    width, _ = board_extent_fraction(recovered, FakeInfo(), 0.2)
    assert 0.30 < width < 0.38


class FakeFisheyeInfo:
    """Realistic OKULO C1W depth-aligned intrinsics."""

    distortion_model = 'kannala_brandt4'
    width = 640
    height = 480
    k = [341.0, 0.0, 325.36, 0.0, 341.0, 239.20, 0.0, 0.0, 1.0]
    d = [-0.0198, 0.0158, -0.1117, 0.2645]


def render_fisheye_view(transform, info):
    """Render the board texture through the fisheye projection model."""
    image = np.full((info.height, info.width, 3), 255, np.uint8)
    k = np.array(info.k, dtype=np.float64).reshape(3, 3)
    d = np.array(info.d, dtype=np.float64).reshape(4, 1)
    dictionary_name, markers = load_board(BOARD_JSON)
    dictionary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dictionary_name))
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    for marker_id, corners in markers.items():
        projected, _ = cv2.fisheye.projectPoints(
            corners.reshape(-1, 1, 3), rvec, transform[:3, 3], k, d)
        destination = projected.reshape(4, 2).astype(np.float32)
        marker = cv2.aruco.drawMarker(dictionary, marker_id, 100)
        source = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32)
        homography = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(marker, homography, (info.width, info.height),
                                     flags=cv2.INTER_NEAREST, borderValue=255)
        image[warped < 128] = 0
    return image


def test_board_pose_recovers_a_fisheye_view():
    # The synthetic renderer warps each marker through a perspective homography,
    # which only approximates a fisheye lens; keep markers large enough that the
    # approximation stays far below the tolerance under test.
    transform = np.eye(4)
    transform[:3, 3] = [0.0, 0.0, 0.35]
    bgr = render_fisheye_view(transform, FakeFisheyeInfo())
    recovered, error, marker_ids, _ = board_pose(bgr, FakeFisheyeInfo(), BOARD_JSON)
    assert len(marker_ids) >= 9
    assert error < 2.0
    assert np.allclose(recovered[:3, 3], [0.0, 0.0, 0.35], atol=0.01)


def test_fisheye_extent_matches_pinhole_scale():
    # fx=341, 0.2 m board at 0.5 m -> 136 px wide, 0.213 of width and 0.284 of height.
    width, height = board_extent_fraction(_transform(0.5), FakeFisheyeInfo(), 0.2)
    assert 0.19 < width < 0.24
    assert 0.26 < height < 0.31


def test_unsupported_distortion_model_is_rejected():
    class OddInfo(FakeInfo):
        distortion_model = 'not_a_real_model'

    with pytest.raises(ValueError, match='Unsupported distortion model'):
        board_extent_fraction(np.eye(4), OddInfo(), 0.2)
