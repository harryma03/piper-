"""Read-only unknown-size red-cylinder observations from synchronized RGB-D."""

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
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32

from piperx_lab.camera_target import decode_color, decode_depth
from piperx_lab.cylinder_geometry import fit_cylinder
from piperx_lab.handeye import camera_model, project_points
from piperx_lab.rgbd_geometry import aligned_frames_valid, color_region_points
from piperx_lab.transforms import TARGET_FRAME


def annotate(color, mask, info, result):
    image = color.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    reference = np.array(result['axis_reference_xyz_m'])
    axis = np.array(result['axis_direction'])
    span = result['observed_red_span_m']
    kind, k, d = camera_model(info)
    pixels = project_points(np.array([reference - axis * span / 2,
                                     reference + axis * span / 2]),
                            np.zeros(3), np.zeros(3), kind, k, d)
    if np.all(np.isfinite(pixels)):
        a, b = [tuple(np.clip(np.rint(v), -10000, 10000).astype(int)) for v in pixels]
        cv2.line(image, a, b, (255, 255, 0), 2)
    lines = [f'cylinder fit: diameter {result["diameter_m"] * 1000:.1f} mm',
             f'visible red span {span * 1000:.1f} mm; full height unknown',
             f'axis reference: {reference[0]:.3f} {reference[1]:.3f} {reference[2]:.3f} m']
    cv2.rectangle(image, (5, 5), (color.shape[1] - 5, 83), (25, 35, 50), -1)
    for row, label in enumerate(lines):
        cv2.putText(image, label, (10, 26 + row * 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return image


class CylinderTarget(Node):
    def __init__(self, samples, save_frame):
        super().__init__('piperx_cylinder_target')
        self.samples_required, self.save_frame = samples, save_frame
        self.samples = []
        self.range_m = None
        self.frames = OrderedDict()
        self.last_attempt = 0.0
        self.reference_pub = self.create_publisher(
            PointStamped, '/piperx/cylinder_axis_reference_camera', 10)
        self.axis_pub = self.create_publisher(
            Vector3Stamped, '/piperx/cylinder_axis_camera', 10)
        for kind, message_type, topic in (
                ('color', Image, '/camera/color/aligned'),
                ('depth', Image, '/camera/depth'),
                ('info', CameraInfo, '/camera/info/depth')):
            self.create_subscription(message_type, topic,
                                     lambda msg, k=kind: self.receive(k, msg), 10)
        self.create_subscription(Float32, '/camera/depth_range', self.on_range, 10)

    def on_range(self, msg):
        if math.isfinite(msg.data) and msg.data > 0:
            self.range_m = float(msg.data)

    def receive(self, kind, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        bundle = self.frames.setdefault(key, {})
        bundle[kind] = msg
        while len(self.frames) > 6:
            self.frames.popitem(last=False)
        if not {'color', 'depth', 'info'} <= bundle.keys():
            return
        del self.frames[key]
        if self.range_m is None or time.monotonic() - self.last_attempt < 1:
            return
        self.last_attempt = time.monotonic()
        if not aligned_frames_valid(bundle):
            self.get_logger().warn('Unexpected RGB-D optical frame; discarding observation')
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - (key[0] + key[1] * 1e-9)
        if not math.isfinite(age) or abs(age) > 0.5:
            self.get_logger().warn('Stale RGB-D observation; waiting for fresh data')
            return
        try:
            color, depth = decode_color(bundle['color']), decode_depth(bundle['depth'])
            points, mask = color_region_points(color, depth, bundle['info'], self.range_m)
            result = fit_cylinder(points)
        except ValueError as exc:
            self.get_logger().warn(f'No observable cylinder: {exc}')
            return
        result.update(frame_id=TARGET_FRAME, stamp={'sec': key[0], 'nanosec': key[1]})
        point = PointStamped()
        point.header = bundle['depth'].header
        point.point.x, point.point.y, point.point.z = result['axis_reference_xyz_m']
        direction = Vector3Stamped()
        direction.header = point.header
        direction.vector.x, direction.vector.y, direction.vector.z = result['axis_direction']
        self.reference_pub.publish(point)
        self.axis_pub.publish(direction)
        self.samples.append(result)
        if self.save_frame is not None:
            self.save_frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.save_frame), annotate(color, mask, bundle['info'], result)):
                self.get_logger().error(f'Could not save {self.save_frame}')
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=25)
    parser.add_argument('--out', type=Path, default=Path('records/cylinder_real/observation.json'))
    parser.add_argument('--save-frame', type=Path)
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:] if argv is None else argv)
    if args.samples < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--samples and --timeout must be positive')
    rclpy.init()
    node = CylinderTarget(args.samples, args.save_frame)
    end = time.monotonic() + args.timeout
    interrupted = False
    try:
        while rclpy.ok() and len(node.samples) < args.samples and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        result = {'complete': len(node.samples) >= args.samples,
                  'samples': node.samples,
                  'measurement_scope': 'visible red surface, no known size or upright prior',
                  'independently_validated': False}
        if node.samples:
            diameters = np.array([v['diameter_m'] for v in node.samples])
            result['diameter_mean_m'] = float(diameters.mean())
            result['diameter_std_m'] = float(diameters.std())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f'Saved {args.out}; complete={result["complete"]}', flush=True)
    return 130 if interrupted else (0 if result['complete'] else 1)


if __name__ == '__main__':
    sys.exit(main())
