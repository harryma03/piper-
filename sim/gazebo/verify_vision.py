"""Check published vision points against Gazebo's independent cube geometry.

Run while start_gazebo_vision.sh is active, in the same isolated ROS domain.
This moves only the Gazebo model and its /set_joint_trajectory interface.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


CASES = [
    ('center', [0.0, 1.8, -1.65, 0.0, 0.0, 0.0], [0.30, 0.0, 0.020], 0.0),
    ('offset', [0.0, 1.8, -1.65, 0.0, 0.0, 0.0], [0.28, 0.035, 0.020], 0.0),
    ('wrist_rotated', [0.08, 1.8, -1.50, 0.0, 0.0, 0.20], [0.30, 0.0, 0.020], 0.0),
    ('cube_yaw_20', [0.0, 1.8, -1.65, 0.0, 0.0, 0.0], [0.30, 0.0, 0.020], math.radians(20)),
]


def pose_matrix(pose):
    q, p = pose.orientation, pose.position
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def cube_surface_distance(point, half_size=0.020):
    delta = np.abs(point) - half_size
    if np.any(delta > 0):
        return float(np.linalg.norm(np.maximum(delta, 0)))
    return float(-np.max(delta))


def square_yaw_error(a, b):
    """Smallest angular distance when rotations by 90 degrees are equivalent."""
    return abs((a - b + math.pi / 4) % (math.pi / 2) - math.pi / 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('records/gazebo_vision/verification.json'))
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--max-surface-error', type=float, default=0.005,
                        help='Simulation-only acceptance threshold in metres')
    parser.add_argument('--max-center-error', type=float, default=0.005,
                        help='Maximum cube centre error in metres')
    parser.add_argument('--max-yaw-error-deg', type=float, default=10.0,
                        help='Maximum cube yaw error modulo 90 degrees')
    args = parser.parse_args(remove_ros_args(args=sys.argv)[1:])
    if (args.samples < 1 or args.max_surface_error <= 0 or args.max_center_error <= 0
            or args.max_yaw_error_deg <= 0):
        parser.error('Samples and surface-error limit must be positive')
    rclpy.init()
    node = Node('piperx_gazebo_vision_verifier')
    states, joints, points, poses = {}, {}, [], []

    def on_states(msg):
        states.update(zip(msg.name, msg.pose))

    def on_joints(msg):
        joints.update(zip(msg.name, msg.position))

    node.create_subscription(ModelStates, '/gazebo/model_states', on_states, 10)
    node.create_subscription(JointState, '/control/joint_states', on_joints, 10)
    node.create_subscription(PointStamped, '/piperx/target_base', points.append, 10)
    node.create_subscription(PoseStamped, '/piperx/cube_pose_base', poses.append, 10)
    publisher = node.create_publisher(JointTrajectory, '/set_joint_trajectory', 10)
    client = node.create_client(SetEntityState, '/gazebo/set_entity_state')

    def wait_for(predicate, timeout=20, description='Gazebo vision data'):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(f'Timed out waiting for {description}; last joints={joints}')

    result = {'robot': 'piper', 'mode': 'sim', 'position_kind': 'visible_surface',
              'cube_size_m': 0.04, 'max_surface_error_m': args.max_surface_error,
              'max_center_error_m': args.max_center_error,
              'max_yaw_error_deg': args.max_yaw_error_deg,
              'cases': [], 'passed': False}
    try:
        if not client.wait_for_service(timeout_sec=10):
            raise TimeoutError('Gazebo state service missing; check isolated ROS_DOMAIN_ID')
        wait_for(lambda: all(name in states for name in ('piper_vision', 'red_target'))
                 and publisher.get_subscription_count() > 0)
        for label, pose, target_position, target_yaw in CASES:
            print(f'Checking {label}', flush=True)
            request = SetEntityState.Request()
            state = EntityState()
            state.name, state.reference_frame = 'red_target', 'world'
            state.pose.position.x, state.pose.position.y, state.pose.position.z = target_position
            state.pose.orientation.z = math.sin(target_yaw / 2)
            state.pose.orientation.w = math.cos(target_yaw / 2)
            request.state = state
            future = client.call_async(request)
            wait_for(future.done, description=f'{label} target move response')
            if not future.result().success:
                raise RuntimeError('Gazebo rejected moving its red test target')
            trajectory = JointTrajectory()
            trajectory.header.frame_id = 'base_link'
            trajectory.joint_names = [f'joint{i}' for i in range(1, 7)]
            point = JointTrajectoryPoint()
            point.positions, point.time_from_start.sec = pose, 0
            trajectory.points = [point]
            publisher.publish(trajectory)
            wait_for(lambda: all(abs(joints.get(name, float('inf')) - angle) < 0.02
                                 for name, angle in zip(trajectory.joint_names, pose)),
                     description=f'{label} joint pose')
            # Let fresh camera frames arrive after the world/arm state settles.
            settled = time.monotonic() + 1.0
            wait_for(lambda: time.monotonic() >= settled)
            points.clear()
            poses.clear()
            wait_for(lambda: len(points) >= args.samples and len(poses) >= args.samples,
                     timeout=args.samples * 6 + 15,
                     description=f'{label} base-frame target and cube-pose samples')
            cube_from_base = np.linalg.inv(pose_matrix(states['red_target'])) @ pose_matrix(states['piper_vision'])
            true_base_yaw = Rotation.from_matrix(
                np.linalg.inv(cube_from_base)[:3, :3]).as_euler('xyz')[2]
            samples = []
            for message in points[:args.samples]:
                if message.header.frame_id != 'base_link':
                    raise ValueError('Expected base_link vision output')
                point_base = np.array([message.point.x, message.point.y, message.point.z])
                point_cube = (cube_from_base @ np.r_[point_base, 1])[:3]
                error = cube_surface_distance(point_cube)
                samples.append({'stamp': {'sec': message.header.stamp.sec,
                                           'nanosec': message.header.stamp.nanosec},
                                'base_xyz_m': point_base.tolist(),
                                'cube_local_xyz_m': point_cube.tolist(),
                                'surface_error_m': error})
            max_error = max(sample['surface_error_m'] for sample in samples)
            pose_samples = []
            for message in poses[:args.samples]:
                if message.header.frame_id != 'base_link':
                    raise ValueError('Expected base_link cube pose')
                p = message.pose.position
                center_base = np.array([p.x, p.y, p.z])
                center_cube = (cube_from_base @ np.r_[center_base, 1])[:3]
                center_error = float(np.linalg.norm(center_cube))
                q = message.pose.orientation
                observed_yaw = math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)
                yaw_error_deg = math.degrees(square_yaw_error(observed_yaw, true_base_yaw))
                pose_samples.append({'stamp': {'sec': message.header.stamp.sec,
                                                'nanosec': message.header.stamp.nanosec},
                                     'center_base_xyz_m': center_base.tolist(),
                                     'center_cube_xyz_m': center_cube.tolist(),
                                     'center_error_m': center_error,
                                     'yaw_error_mod_90_deg': yaw_error_deg})
            max_center_error = max(sample['center_error_m'] for sample in pose_samples)
            max_yaw_error = max(sample['yaw_error_mod_90_deg'] for sample in pose_samples)
            result['cases'].append({'name': label, 'joint_positions_rad': pose,
                                    'target_center_world_m': target_position,
                                    'target_yaw_rad': target_yaw,
                                    'max_surface_error_m': max_error,
                                    'max_center_error_m': max_center_error,
                                    'max_yaw_error_mod_90_deg': max_yaw_error,
                                    'samples': samples, 'cube_pose_samples': pose_samples})
            print(f'{label}: {len(samples)} points, surface {max_error * 1000:.2f} mm, '
                  f'centre {max_center_error * 1000:.2f} mm, '
                  f'yaw {max_yaw_error:.2f} deg max', flush=True)
        result['passed'] = all(case['max_surface_error_m'] <= args.max_surface_error
                               and case['max_center_error_m'] <= args.max_center_error
                               and case['max_yaw_error_mod_90_deg'] <= args.max_yaw_error_deg
                               for case in result['cases'])
    except (TimeoutError, ValueError, RuntimeError) as exc:
        result['error'] = str(exc)
        print(str(exc), file=sys.stderr)
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f'Saved {args.out}; passed={result["passed"]}', flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
