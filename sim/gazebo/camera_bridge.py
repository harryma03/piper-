"""Convert Gazebo's ideal RGB-D sensor to the Okulo topics used by vision.

Gazebo publishes optical-Z depth in metres as 32FC1. Vidu's wrapper publishes
optical-Z depth scaled into 16UC1 plus a range topic. This bridge only changes
the transport representation; it does not emulate ToF noise or lens distortion.
"""

from collections import OrderedDict
from copy import deepcopy
import argparse
import math
import sys

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32


DEPTH_RANGE_M = 3.0


def encode_depth(source: Image, range_m: float) -> Image:
    """Encode finite Gazebo optical-Z metres like Vidu's scaled 16UC1."""
    if source.encoding != '32FC1' or source.step < source.width * 4 or source.step % 4:
        raise ValueError(f'Unexpected Gazebo depth format: {source.encoding}')
    if not math.isfinite(range_m) or range_m <= 0:
        raise ValueError('Depth range must be positive')
    dtype = '>f4' if source.is_bigendian else '<f4'
    raw = np.frombuffer(source.data, dtype=dtype)
    if raw.size != source.height * source.step // 4:
        raise ValueError('Gazebo depth buffer size mismatch')
    depth = raw.reshape(source.height, source.step // 4)[:, :source.width]
    valid = np.isfinite(depth) & (depth > 0) & (depth <= range_m)
    scaled = np.zeros(depth.shape, np.uint16)
    scaled[valid] = np.maximum(1, np.rint(depth[valid] * 65535.0 / range_m)).astype(np.uint16)
    out = Image()
    out.header = source.header
    out.height, out.width = source.height, source.width
    out.encoding, out.is_bigendian, out.step = '16UC1', 0, source.width * 2
    out.data = scaled.astype('<u2', copy=False).tobytes()
    return out


def encode_color(source: Image) -> Image:
    if source.encoding not in ('rgb8', 'bgr8') or source.step < source.width * 3:
        raise ValueError(f'Unexpected Gazebo color format: {source.encoding}')
    raw = np.frombuffer(source.data, np.uint8)
    if raw.size != source.height * source.step:
        raise ValueError('Gazebo color buffer size mismatch')
    pixels = raw.reshape(source.height, source.step)[:, :source.width * 3]
    if source.encoding == 'rgb8':
        pixels = pixels.reshape(source.height, source.width, 3)[:, :, ::-1].copy()
    out = Image()
    out.header = source.header
    out.height, out.width = source.height, source.width
    out.encoding, out.is_bigendian, out.step = '8UC3', 0, source.width * 3
    out.data = pixels.tobytes()
    return out


class CameraBridge(Node):
    def __init__(self, color_only=False):
        super().__init__('piperx_gazebo_camera_bridge')
        self.color_only = color_only
        self.image_pub = self.create_publisher(Image, '/camera/color/aligned', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/info/depth', 10)
        self.range_pub = self.create_publisher(Float32, '/camera/depth_range', 10)
        self.info = None
        self.frames = OrderedDict()
        self.create_subscription(Image, '/sim/camera/aligned/image_raw',
                                 lambda msg: self._receive('color', msg), 10)
        if not color_only:
            self.create_subscription(Image, '/sim/camera/aligned/depth/image_raw',
                                     lambda msg: self._receive('depth', msg), 10)
        info_topic = ('/sim/camera/aligned/camera_info' if color_only else
                      '/sim/camera/aligned/depth/camera_info')
        self.create_subscription(CameraInfo, info_topic,
                                 self._on_info, 10)

    def _on_info(self, info: CameraInfo):
        self.info = info
        if self.color_only:
            self.info_pub.publish(info)

    def _receive(self, kind: str, source: Image):
        if self.color_only:
            try:
                self.image_pub.publish(encode_color(source))
            except ValueError as exc:
                self.get_logger().error(str(exc))
            return
        stamp = (source.header.stamp.sec, source.header.stamp.nanosec)
        bundle = self.frames.setdefault(stamp, {})
        bundle[kind] = source
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if len(bundle) != 2 or self.info is None:
            return
        del self.frames[stamp]
        if bundle['color'].header.frame_id != bundle['depth'].header.frame_id:
            self.get_logger().error('Gazebo color and depth frame IDs disagree')
            return
        if ((bundle['color'].height, bundle['color'].width) !=
                (bundle['depth'].height, bundle['depth'].width) or
                (self.info.height, self.info.width) !=
                (bundle['depth'].height, bundle['depth'].width)):
            self.get_logger().error('Gazebo RGB, depth and intrinsics dimensions disagree')
            return
        try:
            color = encode_color(bundle['color'])
            depth = encode_depth(bundle['depth'], DEPTH_RANGE_M)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        info = deepcopy(self.info)
        info.header = depth.header
        range_msg = Float32()
        range_msg.data = DEPTH_RANGE_M
        self.range_pub.publish(range_msg)
        self.info_pub.publish(info)
        self.image_pub.publish(color)
        self.depth_pub.publish(depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-only', action='store_true',
                        help='Bridge the original color-only hand-eye demo without waiting for depth')
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    rclpy.init()
    node = CameraBridge(color_only=args.color_only)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
