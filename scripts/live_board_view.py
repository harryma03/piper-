#!/usr/bin/env python3
"""Live calibration view: aligned colour + ArUco detection + framing verdict."""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from piperx_lab.handeye import (board_extent_fraction, board_pose, board_size_m,
                                framing_verdict)
from piperx_lab.camera_target import decode_color

SCALE = 2


class LiveBoardView(Node):
    def __init__(self, board_path):
        super().__init__('live_board_view')
        self.board = Path(board_path)
        self.board_size = board_size_m(board_path)
        self.image = None
        self.info = None
        self.create_subscription(Image, '/camera/color/aligned',
                                 lambda m: setattr(self, 'image', m), 5)
        self.create_subscription(CameraInfo, '/camera/info/depth',
                                 lambda m: setattr(self, 'info', m), 5)
        self.last_print = 0.0

    def render(self, bgr):
        info = self.info
        try:
            transform, rms, ids, annotated = board_pose(bgr, info, self.board)
        except Exception as exc:
            out = bgr.copy()
            cv2.putText(out, 'no board: %s' % str(exc)[:60], (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return out, str(exc)
        width, _ = board_extent_fraction(transform, info, self.board_size)
        distance = float(np.linalg.norm(transform[:3, 3]))
        ok, problems = framing_verdict(width, len(ids), rms, distance)
        colour = (0, 200, 0) if ok else (0, 165, 255)
        cv2.putText(annotated, 'markers %d  dist %.3fm  width %.0f%%  rms %.2fpx'
                    % (len(ids), distance, width * 100, rms),
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        cv2.putText(annotated, 'OK' if ok else problems[0][:58],
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        return annotated, ('ok' if ok else '; '.join(problems))


def main():
    board = (get_package_share_directory('piperx_lab') + '/calibration/board.json')
    rclpy.init()
    node = LiveBoardView(board)
    window = 'Piper calibration view (q to quit)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 640 * SCALE, 480 * SCALE)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.image is None or node.info is None:
                continue
            bgr = decode_color(node.image)
            annotated, status = node.render(bgr)
            shown = cv2.resize(annotated, None, fx=SCALE, fy=SCALE,
                               interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window, shown)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            now = time.monotonic()
            if now - node.last_print > 3.0:
                node.last_print = now
                print('status:', status, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
