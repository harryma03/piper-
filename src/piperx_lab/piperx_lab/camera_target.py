"""Observe a colored target in Okulo's aligned color/depth stream; never moves the arm."""

import argparse
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32


HSV_RANGES = {
    'red': [((0, 100, 60), (10, 255, 255)),
            ((170, 100, 60), (179, 255, 255))],
    'orange': [((11, 100, 60), (24, 255, 255))],
    'yellow': [((25, 100, 60), (39, 255, 255))],
    'green': [((40, 75, 50), (85, 255, 255))],
    'blue': [((90, 75, 50), (130, 255, 255))],
    'purple': [((131, 75, 50), (169, 255, 255))],
}


def _stamp(msg):
    return msg.header.stamp.sec, msg.header.stamp.nanosec


def decode_color(msg: Image) -> np.ndarray:
    if msg.encoding != '8UC3' or msg.is_bigendian or msg.step < msg.width * 3:
        raise ValueError(f'Unexpected aligned color format: {msg.encoding}, step={msg.step}')
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if data.size != msg.height * msg.step:
        raise ValueError('Color buffer size does not match Image metadata')
    return data.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)


def decode_depth(msg: Image) -> np.ndarray:
    if msg.encoding != '16UC1' or msg.step < msg.width * 2 or msg.step % 2:
        raise ValueError(f'Unexpected depth format: {msg.encoding}, step={msg.step}')
    dtype = '>u2' if msg.is_bigendian else '<u2'
    data = np.frombuffer(msg.data, dtype=dtype)
    if data.size != msg.height * msg.step // 2:
        raise ValueError('Depth buffer size does not match Image metadata')
    return data.reshape(msg.height, msg.step // 2)[:, :msg.width]


def detect_colored_point(color: np.ndarray, depth: np.ndarray, info: CameraInfo,
                         range_m: float, target_color: str = 'red', min_area: int = 80):
    """Return camera-frame XYZ in metres plus image measurements, or None."""
    if color.shape[:2] != depth.shape or depth.shape != (info.height, info.width):
        raise ValueError('Aligned color, depth, and depth CameraInfo dimensions disagree')
    if info.distortion_model not in ('brown_conrady', 'plumb_bob'):
        raise ValueError(f'Unsupported depth distortion model: {info.distortion_model!r}')
    k = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
    d = np.asarray(info.d, dtype=np.float64)
    if not np.all(np.isfinite(k)) or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.all(np.isfinite(d)):
        raise ValueError('Invalid live depth camera calibration')
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('Invalid depth range')

    # The Vidu wrapper passes OpenCV BGR bytes through ROS with encoding 8UC3.
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask_color = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_RANGES[target_color]:
        mask_color |= cv2.inRange(hsv, lower, upper)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_color, 8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        return None
    label = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    mask = labels == label
    valid = mask & (depth > 0)
    if int(valid.sum()) < max(20, min_area // 4):
        return None
    ys, xs = np.nonzero(valid)
    u, v = float(np.median(xs)), float(np.median(ys))
    depth_raw = float(np.median(depth[valid]))
    z = depth_raw * range_m / 65535.0
    normalized = cv2.undistortPoints(np.array([[[u, v]]], np.float64), k, d)[0, 0]
    x, y = float(normalized[0] * z), float(normalized[1] * z)
    return {
        'pixel': [u, v],
        'xyz_m': [x, y, z],
        'color_area_px': int(stats[label, cv2.CC_STAT_AREA]),
        'valid_depth_px': int(valid.sum()),
        'depth_raw_median': depth_raw,
    }


def detect_red_point(color: np.ndarray, depth: np.ndarray, info: CameraInfo,
                     range_m: float, min_area: int = 80):
    """Keep the original red-target helper available to existing callers."""
    return detect_colored_point(color, depth, info, range_m, 'red', min_area)


class CameraTarget(Node):
    def __init__(self, range_m: float | None, once: bool, target_color: str,
                 save_frame: Path | None):
        super().__init__('piperx_camera_target')
        self.range_m = range_m
        self.manual_range = range_m is not None
        self.once = once
        self.target_color = target_color
        self.save_frame = save_frame
        self.require_target = save_frame is not None
        self.finished = False
        self.last_report = 0.0
        self.frames = OrderedDict()
        self.pub = self.create_publisher(PointStamped, '/piperx/target_camera', 10)
        self.create_subscription(Image, '/camera/color/aligned',
                                 lambda msg: self._receive('color', msg), 10)
        self.create_subscription(Image, '/camera/depth',
                                 lambda msg: self._receive('depth', msg), 10)
        self.create_subscription(CameraInfo, '/camera/info/depth',
                                 lambda msg: self._receive('info', msg), 10)
        self.create_subscription(Float32, '/camera/depth_range', self._on_range, 10)

    def _on_range(self, msg):
        if not self.manual_range and math.isfinite(msg.data) and msg.data > 0:
            self.range_m = float(msg.data)

    def _receive(self, kind, msg):
        key = _stamp(msg)
        bundle = self.frames.setdefault(key, {})
        bundle[kind] = msg
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if not {'color', 'depth', 'info'} <= bundle.keys():
            return
        del self.frames[key]
        if self.range_m is None:
            return
        now = time.monotonic()
        if not self.once and now - self.last_report < 1.0:
            return
        self.last_report = now
        try:
            color = decode_color(bundle['color'])
            depth = decode_depth(bundle['depth'])
            target = detect_colored_point(color, depth, bundle['info'], self.range_m,
                                          self.target_color)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            self.finished = self.once and not self.require_target
            return
        report = {
            'source': 'live Vidu ROS 2 wrapper',
            'frame_id': bundle['depth'].header.frame_id,
            'stamp': {'sec': bundle['depth'].header.stamp.sec,
                      'nanosec': bundle['depth'].header.stamp.nanosec},
            'image_size': [int(depth.shape[1]), int(depth.shape[0])],
            'depth_range_m': self.range_m,
            'target_color': self.target_color,
            'target': target,
        }
        if target is not None and self.save_frame is not None:
            annotated = color.copy()
            u, v = (round(value) for value in target['pixel'])
            cv2.drawMarker(annotated, (u, v), (255, 255, 255),
                           cv2.MARKER_CROSS, 28, 3)
            cv2.drawMarker(annotated, (u, v), (0, 0, 255),
                           cv2.MARKER_CROSS, 24, 2)
            xyz = target['xyz_m']
            label = f'{self.target_color}: X={xyz[0]:.3f} Y={xyz[1]:.3f} Z={xyz[2]:.3f} m'
            cv2.rectangle(annotated, (8, 8), (min(color.shape[1] - 8, 565), 43),
                          (25, 35, 50), -1)
            cv2.putText(annotated, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.57, (255, 255, 255), 2, cv2.LINE_AA)
            self.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.save_frame), annotated):
                self.get_logger().error(f'Could not save annotated frame: {self.save_frame}')
            else:
                report['annotated_frame'] = str(self.save_frame.resolve())
                self.save_frame = None
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if target is not None:
            point = PointStamped()
            point.header = bundle['depth'].header
            point.point.x, point.point.y, point.point.z = target['xyz_m']
            self.pub.publish(point)
        self.finished = self.once and (not self.require_target or
                                       (target is not None and self.save_frame is None))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only Okulo color-target observer')
    parser.add_argument('--color', choices=tuple(HSV_RANGES), default='red',
                        help='Target color; default red')
    parser.add_argument('--save-frame', type=Path,
                        help='Save the first detected frame with center and XYZ overlay')
    parser.add_argument('--once', action='store_true',
                        help='Report one synced frame and exit; with --save-frame, wait for a target')
    parser.add_argument('--timeout', type=float, default=10.0, help='Seconds to wait with --once')
    parser.add_argument('--depth-range', type=float,
                        help='Override live /camera/depth_range in metres')
    args = parser.parse_args(argv)
    if args.depth_range is not None and (not math.isfinite(args.depth_range) or args.depth_range <= 0):
        parser.error('--depth-range must be positive')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive')
    rclpy.init()
    node = CameraTarget(args.depth_range, args.once, args.color, args.save_frame)
    try:
        deadline = time.monotonic() + args.timeout
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.once and time.monotonic() > deadline:
                pending = 'a detected target' if args.save_frame else 'a synchronized Okulo frame'
                print(f'Timed out waiting for {pending}', file=sys.stderr)
                return 1
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
