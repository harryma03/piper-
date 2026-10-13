"""Move the simulated arm through board views and save hand-eye observations."""

import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


POSES = [
    (0.00, 1.80, -1.50, 0.00, 0.00, 0.00),
    (0.14, 1.80, -1.50, 0.00, 0.00, 0.00),
    (-0.14, 1.80, -1.50, 0.00, 0.00, 0.00),
    (0.00, 1.65, -1.50, 0.00, 0.00, 0.00),
    (0.00, 1.95, -1.50, 0.00, 0.00, 0.00),
    (0.00, 1.80, -1.35, 0.00, 0.00, 0.00),
    (0.00, 1.80, -1.65, 0.00, 0.00, 0.00),
    (0.00, 1.80, -1.50, 0.25, 0.00, 0.00),
    (0.00, 1.80, -1.50, -0.25, 0.00, 0.00),
    (0.00, 1.80, -1.50, 0.00, 0.20, 0.00),
    (0.00, 1.80, -1.50, 0.00, -0.20, 0.00),
    (0.00, 1.80, -1.50, 0.00, 0.00, 0.25),
]


def main():
    rclpy.init()
    node = Node('gazebo_handeye_collector')
    pub = node.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
    successes = 0
    try:
        for index, pose in enumerate(POSES, 1):
            trajectory = JointTrajectory()
            trajectory.header.frame_id = 'base_link'
            trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
            point = JointTrajectoryPoint()
            point.positions = list(pose)
            point.time_from_start.sec = 2
            trajectory.points = [point]
            pub.publish(trajectory)
            time.sleep(3.2)
            command = [
                'ros2', 'run', 'piperx_lab', 'handeye_capture',
                '--mode', 'sim', '--timeout', '6',
                '--out-dir', 'records/gazebo_piper_handeye/samples',
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            print(f'{index:02d} {pose}: {(result.stdout + result.stderr).strip()}', flush=True)
            successes += result.returncode == 0
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f'Captured {successes}/{len(POSES)} simulated poses', flush=True)
    return 0 if successes >= 8 else 1


if __name__ == '__main__':
    sys.exit(main())
