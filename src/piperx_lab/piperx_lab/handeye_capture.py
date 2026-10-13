"""Capture one stationary Piper eye-in-hand observation without moving hardware."""

import argparse
from collections import OrderedDict, deque
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformException, TransformListener

from piperx_lab.camera_target import decode_color
from piperx_lab.handeye import board_pose, transform_from_ros
from piperx_lab.model import ARM_JOINTS
from piperx_lab.workspace import workspace_root


# Camera streams selectable for calibration. The 3D target published by
# camera_target lives in camera_depth_frame, so the default source keeps the
# hand-eye result in exactly that frame instead of the raw color frame.
# The Okulo SDK warps the colour image onto the depth sensor geometry but labels
# it 'camera_aligned_color_frame'; it shares the depth optical frame and the
# depth intrinsics, so samples are recorded in the canonical depth frame that
# camera_target publishes.
CAMERA_SOURCES = {
    'depth': ('/camera/color/aligned', '/camera/info/depth', 'camera_depth_frame',
              ('camera_depth_frame', 'camera_aligned_color_frame')),
    'color': ('/camera/color/origin', '/camera/info/color', 'camera_color_frame',
              ('camera_color_frame',)),
}


class HandeyeCapture(Node):
    def __init__(self, state_topic: str, image_topic: str, info_topic: str):
        super().__init__('piperx_handeye_capture')
        self.frames = OrderedDict()
        self.pending = None
        self.joint_history = deque(maxlen=200)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, image_topic,
                                 lambda msg: self._receive('image', msg), 10)
        self.create_subscription(CameraInfo, info_topic,
                                 lambda msg: self._receive('info', msg), 10)
        self.create_subscription(JointState, state_topic, self._on_joints, 10)

    def _receive(self, kind, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = msg
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if 'image' in bundle and 'info' in bundle:
            self.pending = bundle['image'], bundle['info']
            del self.frames[stamp]

    def _on_joints(self, msg):
        values = dict(zip(msg.name, msg.position))
        if all(name in values and math.isfinite(values[name]) for name in ARM_JOINTS):
            self.joint_history.append((time.monotonic(),
                                       np.array([values[name] for name in ARM_JOINTS])))

    def stationary_joints(self):
        if not self.joint_history:
            return None
        now = time.monotonic()
        recent = [(t, q) for t, q in self.joint_history if now - t <= 0.8]
        if len(recent) < 3 or now - recent[-1][0] > 0.3 or recent[-1][0] - recent[0][0] < 0.4:
            return None
        latest = recent[-1][1]
        if max(float(np.max(np.abs(q - latest))) for _, q in recent) > 0.004:
            return None
        return latest


def default_board() -> Path:
    try:
        return workspace_root() / 'calibration/board.json'
    except FileNotFoundError:
        return Path(get_package_share_directory('piperx_lab')) / 'calibration/board.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Capture one fixed-board hand-eye observation')
    parser.add_argument('--mode', choices=('real', 'sim'), default='real')
    parser.add_argument('--board', type=Path)
    parser.add_argument('--out-dir', type=Path,
                        help='Sample directory; defaults to <workspace>/calibration/samples')
    parser.add_argument('--base-frame', default='base_link')
    parser.add_argument('--wrist-frame', choices=('link6',), default='link6')
    parser.add_argument('--camera-source', choices=tuple(CAMERA_SOURCES), default='depth',
                        help='depth: aligned color + depth info (camera_depth_frame, '
                             'matches camera_target); color: raw 1920x1080 color')
    parser.add_argument('--timeout', type=float, default=45.0)
    parser.add_argument('--max-reprojection', type=float, default=2.0,
                        help='Maximum board reprojection RMS in pixels')
    args = parser.parse_args(argv)
    if args.out_dir is None:
        try:
            args.out_dir = workspace_root() / 'calibration/samples'
        except FileNotFoundError as exc:
            parser.error(str(exc))
    if args.timeout <= 0 or args.max_reprojection <= 0:
        parser.error('Timeout and reprojection limit must be positive')
    board = args.board or default_board()
    if not board.is_file():
        parser.error(f'Board description missing: {board}')
    state_topic = '/feedback/joint_states' if args.mode == 'real' else '/control/joint_states'
    image_topic, info_topic, expected_frame, accepted_frames = CAMERA_SOURCES[args.camera_source]
    rclpy.init()
    node = HandeyeCapture(state_topic, image_topic, info_topic)
    last_message = 0.0
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.pending is None:
                continue
            image, info = node.pending
            node.pending = None
            joints = node.stationary_joints()
            if joints is None:
                reason = 'No stable fresh six-joint state yet'
            else:
                try:
                    bgr = decode_color(image)
                    if image.header.frame_id not in accepted_frames:
                        raise ValueError(
                            f'Expected frame {expected_frame}, got {image.header.frame_id!r}')
                    camera_board, error, marker_ids, annotated = board_pose(bgr, info, board)
                    if error > args.max_reprojection:
                        raise ValueError(f'Board reprojection RMS {error:.2f} px exceeds limit')
                    tf = node.tf_buffer.lookup_transform(
                        args.base_frame, args.wrist_frame,
                        Time.from_msg(image.header.stamp))
                    base_wrist = transform_from_ros(tf)
                except (ValueError, TransformException) as exc:
                    reason = str(exc)
                else:
                    args.out_dir.mkdir(parents=True, exist_ok=True)
                    stem = f'{image.header.stamp.sec}_{image.header.stamp.nanosec:09d}'
                    raw_path = args.out_dir / f'{stem}_color.png'
                    overlay_path = args.out_dir / f'{stem}_detected.jpg'
                    json_path = args.out_dir / f'{stem}.json'
                    if not cv2.imwrite(str(raw_path), bgr) or not cv2.imwrite(str(overlay_path), annotated):
                        raise RuntimeError('Could not write captured images')
                    sample = {
                        'robot': 'piper',
                        'camera': ('Gazebo ideal camera' if args.mode == 'sim'
                                   else 'OKULO-C1W-H106/940'),
                        'mode': args.mode,
                        # Paths are relative to this JSON file, so the captured
                        # images remain usable after cloning on another machine.
                        'board_json': os.path.relpath(board.resolve(), json_path.parent.resolve()),
                        'color_image': raw_path.name,
                        'detected_image': overlay_path.name,
                        'stamp': {'sec': image.header.stamp.sec,
                                  'nanosec': image.header.stamp.nanosec},
                        'base_frame': args.base_frame,
                        'wrist_frame': args.wrist_frame,
                        'camera_source': args.camera_source,
                        'camera_frame': expected_frame,
                        'image_frame_id': image.header.frame_id,
                        'joint_positions_rad': dict(zip(ARM_JOINTS, joints.tolist())),
                        'color_info': {'width': info.width, 'height': info.height,
                                       'distortion_model': info.distortion_model,
                                       'k': list(info.k), 'd': list(info.d)},
                        'marker_ids': marker_ids,
                        'reprojection_rms_px': error,
                        'T_base_wrist': base_wrist.tolist(),
                        'T_camera_board': camera_board.tolist(),
                    }
                    json_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + '\n',
                                         encoding='utf-8')
                    print(f'Saved {json_path} ({len(marker_ids)} markers, RMS {error:.2f} px)')
                    return 0
            if time.monotonic() - last_message > 2.0:
                print(f'Waiting: {reason}', file=sys.stderr)
                last_message = time.monotonic()
        print('Timed out waiting for a stationary arm, visible board, and valid TF', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
