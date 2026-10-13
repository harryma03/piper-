"""Check that the ArUco calibration board is placed so the camera can see it.

Read-only helper for the physical step of the hand-eye calibration: it reports
how many markers are visible, how far away the board is, how much of the frame
it fills, and whether that framing is usable. It never moves the arm.
"""

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from piperx_lab.camera_target import decode_color
from piperx_lab.handeye import (board_extent_fraction, board_pose, board_size_m,
                              framing_verdict)


def analyse(bgr: np.ndarray, info: CameraInfo, board_path: Path) -> dict:
    """Detect the board and summarise how it is framed."""
    transform, error, marker_ids, annotated = board_pose(bgr, info, board_path)
    fill_width, fill_height = board_extent_fraction(
        transform, info, board_size_m(board_path))
    distance = float(np.linalg.norm(transform[:3, 3]))
    ok, problems = framing_verdict(fill_width, len(marker_ids), error, distance)
    return {
        'marker_count': len(marker_ids),
        'marker_ids': marker_ids,
        'board_distance_m': round(distance, 4),
        'fill_fraction_width': round(fill_width, 3),
        'fill_fraction_height': round(fill_height, 3),
        'reprojection_rms_px': round(error, 3),
        'ok': ok,
        'problems': problems,
        'annotated': annotated,
    }


class BoardCheck(Node):
    def __init__(self, image_topic: str, info_topic: str, board_path: Path,
                 save_dir: Path | None = None, once: bool = False):
        super().__init__('piperx_board_check')
        self.board_path = board_path
        self.save_dir = save_dir
        self.once = once
        self.finished = False
        self.frames = OrderedDict()
        self.create_subscription(Image, image_topic,
                                 lambda msg: self._receive('image', msg), 10)
        self.create_subscription(CameraInfo, info_topic,
                                 lambda msg: self._receive('info', msg), 10)

    def _receive(self, kind, msg):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = msg
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if 'image' in bundle and 'info' in bundle:
            self._analyse(bundle['image'], bundle['info'])
            del self.frames[stamp]

    def _analyse(self, image, info):
        try:
            bgr = decode_color(image)
            report = analyse(bgr, info, self.board_path)
        except ValueError as exc:
            self.get_logger().warn(f'No usable board yet: {exc}')
            return
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            out = self.save_dir / (
                f'board_check_{image.header.stamp.sec}_'
                f'{image.header.stamp.nanosec:09d}.jpg')
            cv2.imwrite(str(out), report.pop('annotated'))
            report['annotated_image'] = str(out.resolve())
        else:
            report.pop('annotated')
        report['camera_frame'] = image.header.frame_id
        print(json.dumps(report, ensure_ascii=False), flush=True)
        self.finished = self.once


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check ArUco board placement for hand-eye calibration')
    parser.add_argument('--camera-source', choices=('depth', 'color'), default='depth')
    parser.add_argument('--board', type=Path)
    parser.add_argument('--save-image', type=Path,
                        help='Directory to save the annotated board image')
    parser.add_argument('--once', action='store_true', help='Report one frame and exit')
    parser.add_argument('--timeout', type=float, default=15.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    board = args.board or Path(get_package_share_directory('piperx_lab')) / 'calibration/board.json'
    if not board.is_file():
        parser.error(f'Board description missing: {board}')
    image_topic, info_topic = {
        'depth': ('/camera/color/aligned', '/camera/info/depth'),
        'color': ('/camera/color/origin', '/camera/info/color'),
    }[args.camera_source]

    rclpy.init()
    node = BoardCheck(image_topic, info_topic, board, args.save_image, args.once)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.once and time.monotonic() > deadline:
                print('Timed out waiting for a frame with a detectable board', file=sys.stderr)
                return 1
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
