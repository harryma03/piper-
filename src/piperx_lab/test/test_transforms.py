import json
import math
from pathlib import Path

import numpy as np
import pytest

from piperx_lab.transforms import (camera_point_to_base, load_handeye,
                                   rotation_matrix_to_quaternion, transform_points)


def _rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def test_identity_leaves_point_unchanged():
    point = camera_point_to_base(np.eye(4), np.eye(4), [0.1, -0.2, 0.5])
    assert np.allclose(point, [0.1, -0.2, 0.5])


def test_wrist_offset_shifts_point():
    T_wrist_camera = np.eye(4)
    T_wrist_camera[:3, 3] = [0.0, 0.0, 0.03]
    T_base_wrist = np.eye(4)
    T_base_wrist[:3, 3] = [0.2, -0.1, 0.4]
    point = camera_point_to_base(T_wrist_camera, T_base_wrist, [0.0, 0.0, 0.5])
    assert np.allclose(point, [0.2, -0.1, 0.93])


def test_rotation_is_applied():
    T_wrist_camera = np.eye(4)
    T_wrist_camera[:3, :3] = _rx(math.pi)
    point = camera_point_to_base(T_wrist_camera, np.eye(4), [0.0, 0.0, 0.5])
    assert np.allclose(point, [0.0, 0.0, -0.5])


def test_transform_points_handles_arrays():
    moved = transform_points(np.eye(4), [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert moved.shape == (2, 3)
    assert np.allclose(moved[1], [1.0, 2.0, 3.0])


def test_transform_points_rejects_bad_shape():
    with pytest.raises(ValueError):
        transform_points(np.eye(4), [0.0, 1.0])


def test_quaternion_for_pure_x_rotation():
    quaternion = rotation_matrix_to_quaternion(_rx(math.pi / 2.0))
    x, y, z, w = quaternion
    assert math.isclose(x, math.sin(math.pi / 4.0), abs_tol=1e-9)
    assert math.isclose(w, math.cos(math.pi / 4.0), abs_tol=1e-9)
    assert abs(y) < 1e-9 and abs(z) < 1e-9


def _write_handeye(path: Path, **overrides):
    data = {
        'T_wrist_camera': np.eye(4).tolist(),
        'robot': 'piper',
        'mode': 'real',
        'base_frame': 'base_link',
        'wrist_frame': 'link6',
        'camera_frame': 'camera_depth_frame',
        'quality_pass': True,
        'sample_count': 8,
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_load_handeye_accepts_valid_result(tmp_path):
    result = load_handeye(_write_handeye(tmp_path / 'handeye_result.json'))
    assert result['quality_pass'] is True
    assert result['camera_frame'] == 'camera_depth_frame'
    assert np.allclose(result['T_wrist_camera'], np.eye(4))


def test_load_handeye_rejects_wrong_frames(tmp_path):
    path = _write_handeye(tmp_path / 'r.json', wrist_frame='link5')
    with pytest.raises(ValueError, match='link6'):
        load_handeye(path)


def test_load_handeye_rejects_color_frame(tmp_path):
    path = _write_handeye(tmp_path / 'r.json', camera_frame='camera_color_frame')
    with pytest.raises(ValueError, match='camera_depth_frame'):
        load_handeye(path)


def test_load_handeye_rejects_other_robot_or_simulation(tmp_path):
    for fields in ({'robot': 'piper_x'}, {'mode': 'sim'}):
        path = _write_handeye(tmp_path / 'r.json', **fields)
        with pytest.raises(ValueError, match='real Piper'):
            load_handeye(path)
