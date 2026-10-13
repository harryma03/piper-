"""Load a hand-eye result and move points between arm and camera frames.

The wrist frame is always ``link6`` and the base frame is always ``base_link``;
a calibration result for any other pair is rejected rather than silently used.
All transforms follow the ``T_a_b`` convention: ``T_a_b @ p_b = p_a``.
"""

import json
import math
from pathlib import Path

import numpy as np

from piperx_lab.handeye import as_transform

BASE_FRAME = 'base_link'
WRIST_FRAME = 'link6'
TARGET_FRAME = 'camera_depth_frame'


def load_handeye(path: Path) -> dict:
    """Read and validate ``calibration/handeye_result.json``."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('robot') != 'piper' or data.get('mode') != 'real':
        raise ValueError('hand-eye result must come from real Piper samples')
    if 'T_wrist_camera' not in data:
        raise ValueError('hand-eye result is missing T_wrist_camera')
    transform = as_transform(data['T_wrist_camera'])
    base_frame, wrist_frame = data.get('base_frame'), data.get('wrist_frame')
    if base_frame != BASE_FRAME or wrist_frame != WRIST_FRAME:
        raise ValueError(
            f'hand-eye result frames must be {BASE_FRAME}->{WRIST_FRAME}, '
            f'got {base_frame}->{wrist_frame}')
    camera_frame = data.get('camera_frame')
    if camera_frame != TARGET_FRAME:
        raise ValueError(
            f'hand-eye result camera frame must be {TARGET_FRAME}, got {camera_frame!r}; '
            'recalibrate with handeye_capture --camera-source depth')
    return {
        'T_wrist_camera': transform,
        'base_frame': base_frame,
        'wrist_frame': wrist_frame,
        'camera_frame': camera_frame,
        'quality_pass': bool(data.get('quality_pass')),
        'sample_count': int(data.get('sample_count', 0)),
    }


def transform_points(transform: np.ndarray, points) -> np.ndarray:
    """Apply a 4x4 transform to one ``(3,)`` point or an ``(N, 3)`` array."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError('Expected a 4x4 transform')
    array = np.asarray(points, dtype=np.float64)
    single = array.ndim == 1
    array = np.atleast_2d(array)
    if array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError('points must be a finite (N, 3) array')
    homogeneous = np.concatenate([array, np.ones((array.shape[0], 1))], axis=1)
    moved = (matrix @ homogeneous.T).T[:, :3]
    return moved[0] if single else moved


def camera_point_to_base(T_wrist_camera: np.ndarray, T_base_wrist: np.ndarray,
                         point) -> np.ndarray:
    """Map a point in the camera depth frame into ``base_link``."""
    return transform_points(np.asarray(T_base_wrist) @ np.asarray(T_wrist_camera), point)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple:
    """Convert a 3x3 rotation matrix into an ``(x, y, z, w)`` quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError('Expected a 3x3 rotation matrix')
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)
