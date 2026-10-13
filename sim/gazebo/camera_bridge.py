"""Expose Gazebo's ideal camera under the Okulo topics used for calibration.

This bridge only converts RGB byte order to the Vidu wrapper's BGR 8UC3
format. It does not simulate ToF physics or real camera noise.
"""

import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraBridge(Node):
    def __init__(self):
        super().__init__('piperx_gazebo_camera_bridge')
        self.image_pub = self.create_publisher(Image, '/camera/color/aligned', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/info/depth', 10)
        self.create_subscription(Image, '/sim/camera/aligned/image_raw', self.on_image, 10)
        self.create_subscription(CameraInfo, '/sim/camera/aligned/camera_info',
                                 self.on_info, 10)

    def on_image(self, source: Image):
        if source.encoding != 'rgb8' or source.step != source.width * 3:
            self.get_logger().error(f'Unexpected Gazebo image: {source.encoding}')
            return
        out = Image()
        out.header = source.header
        out.height, out.width = source.height, source.width
        out.encoding, out.is_bigendian, out.step = '8UC3', 0, source.step
        pixels = np.frombuffer(source.data, np.uint8).reshape(-1, 3)
        out.data = pixels[:, ::-1].copy().tobytes()
        self.image_pub.publish(out)

    def on_info(self, source: CameraInfo):
        self.info_pub.publish(source)


def main():
    rclpy.init()
    node = CameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
