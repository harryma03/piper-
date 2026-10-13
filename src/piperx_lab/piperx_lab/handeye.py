"""Eye-in-hand board pose and hand-eye calibration for Piper."""

import json
import math
from pathlib import Path

import cv2
import numpy as np


def as_transform(value) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('Expected a finite 4x4 transform')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous transform bottom row')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or np.linalg.det(rotation) < 0.99:
        raise ValueError('Invalid rotation matrix')
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = transform[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ transform[:3, 3]
    return out


def transform_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(translation).reshape(3)
    return out


def transform_from_ros(stamped) -> np.ndarray:
    t = stamped.transform.translation
    q = stamped.transform.rotation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        raise ValueError('Invalid zero quaternion from TF')
    x, y, z, w = quat / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return transform_from_pose(rotation, [t.x, t.y, t.z])


def load_board(path: Path):
    board = json.loads(path.read_text(encoding='utf-8'))
    dictionary_name = board['dictionary']
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f'Unknown ArUco dictionary {dictionary_name}')
    markers = {}
    for marker_id, marker in board['markers'].items():
        corners = np.asarray(marker['corners'], dtype=np.float64)
        if corners.shape != (4, 3):
            raise ValueError(f'Invalid board corners for marker {marker_id}')
        markers[int(marker_id)] = corners
    return dictionary_name, markers


PINHOLE_MODELS = ('brown_conrady', 'plumb_bob')
FISHEYE_MODELS = ('kannala_brandt4', 'equidistant', 'fisheye')


def camera_model(camera_info):
    """Return ``(kind, K, D)`` for a pinhole or fisheye CameraInfo."""
    name = camera_info.distortion_model
    k = np.asarray(camera_info.k, np.float64).reshape(3, 3)
    d = np.asarray(camera_info.d, np.float64)
    if (not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.all(np.isfinite(d))):
        raise ValueError('Invalid live CameraInfo')
    if name in PINHOLE_MODELS:
        return 'pinhole', k, d.reshape(-1)
    if name in FISHEYE_MODELS:
        if d.size < 4:
            raise ValueError(f'Fisheye CameraInfo needs four coefficients, got {d.size}')
        return 'fisheye', k, d[:4].reshape(4, 1)
    raise ValueError(f'Unsupported distortion model: {name!r}')


def undistort_pixels(points, kind: str, k: np.ndarray, d) -> np.ndarray:
    """Map detected pixels onto the ideal pinhole model that shares ``K``."""
    raw = np.asarray(points, np.float64).reshape(-1, 1, 2)
    if kind == 'fisheye':
        out = cv2.fisheye.undistortPoints(raw, k, d, P=k)
    else:
        out = cv2.undistortPoints(raw, k, d, P=k)
    return np.asarray(out, np.float64).reshape(-1, 2)


def project_points(points, rvec, tvec, kind: str, k: np.ndarray, d) -> np.ndarray:
    """Project 3D points with the camera model that produced ``K``/``D``."""
    raw = np.asarray(points, np.float64).reshape(-1, 1, 3)
    if kind == 'fisheye':
        out, _ = cv2.fisheye.projectPoints(raw, rvec, tvec, k, d)
    else:
        out, _ = cv2.projectPoints(raw, rvec, tvec, k, d)
    return np.asarray(out, np.float64).reshape(-1, 2)


def _draw_axes(image, rvec, tvec, kind: str, k, d, length: float = 0.05) -> None:
    axes = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0],
                     [0.0, length, 0.0], [0.0, 0.0, length]])
    pixels = project_points(axes, rvec, tvec, kind, k, d)
    origin = tuple(np.round(pixels[0]).astype(int))
    for index, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        tip = tuple(np.round(pixels[index + 1]).astype(int))
        cv2.line(image, origin, tip, colour, 2)


def board_pose(bgr: np.ndarray, camera_info, board_path: Path):
    """Return T_camera_board, reprojection RMS, count, and annotated image.

    Both pinhole and fisheye (Kannala-Brandt) CameraInfo are accepted. Fisheye
    pixels are mapped onto the ideal pinhole model that shares the same ``K``
    so board pose stays a standard perspective solvePnP problem.
    """
    if bgr.shape[:2] != (camera_info.height, camera_info.width):
        raise ValueError('Color frame dimensions do not match CameraInfo')
    kind, k, d = camera_model(camera_info)
    dictionary_name, markers = load_board(board_path)
    dictionary = cv2.aruco.Dictionary_get(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters_create()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        raise ValueError('No calibration markers detected')
    object_points, detected_points, used_ids = [], [], []
    for image_corners, marker_id in zip(corners, ids.reshape(-1)):
        marker_id = int(marker_id)
        if marker_id in markers:
            object_points.extend(markers[marker_id])
            detected_points.extend(image_corners.reshape(4, 2))
            used_ids.append(marker_id)
    if len(used_ids) < 4:
        raise ValueError(f'Need at least 4 board markers; found {len(used_ids)}')
    object_points = np.asarray(object_points, np.float64)
    image_points = undistort_pixels(detected_points, kind, k, d)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, k, None,
        iterationsCount=500, reprojectionError=2.0, confidence=0.99,
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inliers is None or len(inliers) < 16:
        raise ValueError('Board solvePnP failed or found fewer than 16 inlier corners')
    if hasattr(cv2, 'solvePnPRefineLM'):
        selected = inliers.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points[selected], image_points[selected], k, None, rvec, tvec)
    if float(tvec[2]) <= 0:
        raise ValueError('Board pose is behind the camera')
    projected = project_points(object_points, rvec, tvec, 'pinhole', k, None)
    error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
    rotation, _ = cv2.Rodrigues(rvec)
    annotated = bgr.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    _draw_axes(annotated, rvec, tvec, kind, k, d)
    return transform_from_pose(rotation, tvec), error, sorted(used_ids), annotated


def board_size_m(path: Path) -> float:
    """Read the physical board edge length in metres from a board description."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    size = float(data['board_size'])
    if not math.isfinite(size) or size <= 0.0:
        raise ValueError('Invalid board_size in board description')
    return size


def board_extent_fraction(transform: np.ndarray, camera_info, board_size: float) -> tuple:
    """Fraction of the image width/height covered by the board's outer corners."""
    kind, k, d = camera_model(camera_info)
    corners = np.array([[0.0, 0.0, 0.0], [board_size, 0.0, 0.0],
                        [board_size, board_size, 0.0], [0.0, board_size, 0.0]])
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    pixels = project_points(corners, rvec, transform[:3, 3], kind, k, d)
    width = float(pixels[:, 0].max() - pixels[:, 0].min()) / camera_info.width
    height = float(pixels[:, 1].max() - pixels[:, 1].min()) / camera_info.height
    return width, height


def framing_verdict(fill_width: float, marker_count: int, rms_px: float,
                    distance_m: float, min_fill: float = 0.20,
                    max_fill: float = 0.70) -> tuple:
    """Judge whether one board view is usable for hand-eye calibration."""
    problems = []
    if marker_count < 9:
        problems.append(f'only {marker_count} markers visible; move so more are in view')
    if fill_width < min_fill:
        problems.append(f'board fills only {fill_width * 100:.0f}% of frame width; move closer')
    if fill_width > max_fill:
        problems.append(f'board fills {fill_width * 100:.0f}% of frame width; move further away')
    if not math.isfinite(distance_m) or distance_m <= 0.05:
        problems.append(f'board distance {distance_m:.3f} m looks wrong')
    if not math.isfinite(rms_px) or rms_px > 2.0:
        problems.append(f'reprojection RMS {rms_px:.2f} px is high; check focus/glare')
    return (not problems), problems


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    c = max(-1.0, min(1.0, (np.trace(a.T @ b) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def solve_eye_in_hand(samples: list[dict]) -> dict:
    """OpenCV inputs: gripper-to-base and fixed-target-to-camera transforms."""
    if len(samples) < 5:
        raise ValueError('Need at least five distinct stationary board observations')
    poses = [as_transform(item['T_base_wrist']) for item in samples]
    board_poses = [as_transform(item['T_camera_board']) for item in samples]
    spread = max(_rotation_angle(a[:3, :3], b[:3, :3])
                 for a in poses for b in poses)
    if spread < 15.0:
        raise ValueError(f'Arm orientation spread only {spread:.1f} deg; collect varied wrist poses')
    rotation, translation = cv2.calibrateHandEye(
        [pose[:3, :3] for pose in poses], [pose[:3, 3] for pose in poses],
        [pose[:3, :3] for pose in board_poses], [pose[:3, 3] for pose in board_poses],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    handeye = as_transform(transform_from_pose(rotation, translation))
    base_board = [arm @ handeye @ camera_board
                  for arm, camera_board in zip(poses, board_poses)]
    center = np.mean([pose[:3, 3] for pose in base_board], axis=0)
    translation_errors = [float(np.linalg.norm(pose[:3, 3] - center)) for pose in base_board]
    rotation_errors = [_rotation_angle(base_board[0][:3, :3], pose[:3, :3])
                       for pose in base_board]
    return {
        'method': 'OpenCV PARK eye-in-hand',
        'sample_count': len(samples),
        'arm_orientation_spread_deg': spread,
        'T_wrist_camera': handeye.tolist(),
        'fixed_board_position_spread_max_m': max(translation_errors),
        'fixed_board_rotation_spread_max_deg': max(rotation_errors),
    }
