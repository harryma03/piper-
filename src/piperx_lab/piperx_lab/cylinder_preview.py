"""Visualize cylinder grasp hypotheses and request MoveIt plans only.

This node has no execution action client and publishes no control commands.
Plans assume the gripper has been opened; the real gripper is never commanded.
"""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import (CollisionObject, Constraints, DisplayTrajectory,
                             JointConstraint, RobotState)
from moveit_msgs.srv import (ApplyPlanningScene, GetCartesianPath, GetMotionPlan,
                             GetPositionIK)
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from piperx_lab.cylinder_grasp import grasp_candidates, model_tip_transform, full_cylinder_geometry
from piperx_lab.handeye import transform_from_ros
from piperx_lab.model import installed_gripper_model_path, read_gripper_limit


def pose_from_matrix(matrix):
    matrix = np.asarray(matrix)
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in matrix[:3, 3])
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in q)
    return pose


def rotation_with_z(axis):
    z = np.asarray(axis, np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(np.eye(3)[np.argmin(np.abs(z))], z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


class CylinderPreview(Node):
    def __init__(self):
        super().__init__('piperx_cylinder_preview')
        self.observation = None
        self.joints = None
        self._service_clients = {}
        self.create_subscription(String, '/piperx/cylinder_observation', self.on_observation, 10)
        self.create_subscription(JointState, '/feedback/joint_states', self.on_joints, 10)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.marker_pub = self.create_publisher(MarkerArray, '/piperx/cylinder_preview_markers', latched)
        self.path_pub = self.create_publisher(DisplayTrajectory, '/piperx/cylinder_preview_trajectory', latched)

    def on_observation(self, msg):
        try:
            sample = json.loads(msg.data)
            if sample.get('base', {}).get('frame_id') == 'base_link':
                self.observation = sample
        except (ValueError, KeyError):
            pass

    def on_joints(self, msg):
        if all(f'joint{i}' in msg.name for i in range(1, 7)):
            self.joints = msg

    def wait(self, predicate, timeout=15):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError('Timed out waiting for fresh cylinder, robot state or service response')

    def age(self, stamp):
        seconds = stamp['sec'] + stamp['nanosec'] * 1e-9 if isinstance(stamp, dict) else stamp.sec + stamp.nanosec * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9 - seconds

    def fresh(self):
        return (self.observation is not None and self.joints is not None and
                abs(self.age(self.observation['stamp'])) < 1.0 and
                abs(self.age(self.joints.header.stamp)) < 0.5)

    def check_snapshot(self, sample, initial):
        # A 1 Hz perception stream may cross its next-frame boundary during IK.
        # Wait for a genuinely fresh sample instead of relaxing the age limit.
        if not self.fresh():
            self.wait(self.fresh, 3)
        if not self.fresh():
            raise ValueError('Current camera or joint feedback is stale; discard preview')
        actual = dict(zip(self.joints.name, self.joints.position))
        start = dict(zip(initial.joint_state.name, initial.joint_state.position))
        if any(abs(actual[f'joint{i}'] - start[f'joint{i}']) > 0.03 for i in range(1, 7)):
            raise ValueError('The real arm moved during planning; discard preview')
        before, after = sample['base'], self.observation['base']
        if (np.linalg.norm(np.array(before['axis_reference_xyz_m']) - after['axis_reference_xyz_m']) > 0.01
                or abs(np.dot(before['axis_direction'], after['axis_direction'])) < math.cos(math.radians(10))
                or abs(before['diameter_m'] - after['diameter_m']) > 0.005):
            raise ValueError('The observed cylinder changed during planning; discard preview')

    def service(self, service_type, name, request, timeout=8):
        if name not in self._service_clients:
            self._service_clients[name] = self.create_client(service_type, name)
        client = self._service_clients[name]
        if not client.wait_for_service(timeout_sec=2):
            raise RuntimeError(f'MoveIt service unavailable: {name}')
        future = client.call_async(request)
        self.wait(future.done, timeout)
        response = future.result()
        if response is None:
            raise RuntimeError(f'No response from {name}')
        return response

    def update_scene(self, base=None):
        request = ApplyPlanningScene.Request()
        request.scene.is_diff = True
        request.scene.robot_state.is_diff = True
        for name in ('support', 'observed_band'):
            obj = CollisionObject()
            obj.id = 'piper_cylinder_preview_' + name
            obj.header.frame_id = 'base_link'
            obj.operation = CollisionObject.REMOVE if base is None else CollisionObject.ADD
            if base is not None:
                primitive = SolidPrimitive()
                matrix = np.eye(4)
                if name == 'support':
                    patch = base['support_patch']
                    normal = np.array(patch['normal'])
                    primitive.type = SolidPrimitive.BOX
                    primitive.dimensions = patch['patch_size_m']
                    matrix[:3, 3] = np.array(patch['surface_point_m']) - normal * 0.01
                    matrix[:3, :3] = rotation_with_z(normal)
                else:
                    primitive.type = SolidPrimitive.CYLINDER
                    full = base.get('full_geometry')
                    primitive.dimensions = ([full['height_m'], full['diameter_m'] / 2 + 0.003]
                                            if full else [base['observed_red_span_m'] + 0.010,
                                                          base['diameter_m'] / 2 + 0.003])
                    matrix[:3, 3] = full['center_xyz_m'] if full else base['axis_reference_xyz_m']
                    matrix[:3, :3] = rotation_with_z(full['axis_direction'] if full else base['axis_direction'])
                obj.primitives = [primitive]
                obj.primitive_poses = [pose_from_matrix(matrix)]
            request.scene.world.collision_objects.append(obj)
        if not self.service(ApplyPlanningScene, '/apply_planning_scene', request).success:
            raise RuntimeError('MoveIt rejected the preview collision geometry')

    def markers(self, base, candidates, selected=None):
        output = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)

        def add(kind, color):
            item = Marker()
            item.header.frame_id = 'base_link'
            item.ns = 'cylinder_preview'
            item.id = len(output.markers)
            item.type, item.action = kind, Marker.ADD
            item.pose.orientation.w = 1.0
            item.color.r, item.color.g, item.color.b, item.color.a = color
            output.markers.append(item)
            return item

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_with_z(base['axis_direction'])
        matrix[:3, 3] = base['axis_reference_xyz_m']
        cylinder = add(Marker.CYLINDER, (1.0, 0.1, 0.1, 0.5))
        cylinder.pose = pose_from_matrix(matrix)
        cylinder.scale.x = cylinder.scale.y = base['diameter_m']
        cylinder.scale.z = base['observed_red_span_m']
        for candidate in candidates:
            tip = np.array(candidate['T_base_tip'])
            width = candidate['opening_m']
            item = add(Marker.LINE_LIST, (0.0, 1.0, 0.1, 1.0) if candidate['id'] == selected
                       else (0.1, 0.6, 1.0, 0.35))
            item.scale.x = 0.003
            lines = []
            for side in (-1, 1):
                for x in (-0.028, 0.028):
                    lines.append(([x, side * width / 2, 0], [x, side * width / 2, -0.076]))
            lines += [([0, -width / 2, -0.076], [0, width / 2, -0.076]),
                      ([0, 0, -0.10], [0, 0, 0])]
            for line in lines:
                for vertex in line:
                    p = (tip @ np.r_[vertex, 1])[:3]
                    item.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        text = add(Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0, 1.0))
        text.pose.position.x, text.pose.position.y, text.pose.position.z = (
            float(v) for v in np.array(base['axis_reference_xyz_m']) + [0, 0, 0.15])
        text.scale.z = 0.025
        text.text = 'CYLINDER PREVIEW ONLY\nPartial scene / model fingertip reference'
        self.marker_pub.publish(output)

    def ik(self, matrix, state, avoid_collisions=True):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'arm'
        request.ik_request.ik_link_name = 'tcp_link'
        request.ik_request.robot_state = state
        request.ik_request.avoid_collisions = avoid_collisions
        request.ik_request.timeout.sec = 1
        request.ik_request.pose_stamped.header.frame_id = 'base_link'
        request.ik_request.pose_stamped.pose = pose_from_matrix(matrix)
        return self.service(GetPositionIK, '/compute_ik', request, 4)

    def plan_candidate(self, candidate, initial):
        ik = self.ik(candidate['T_base_pregrasp_tcp'], initial)
        if ik.error_code.val != 1:
            diagnostic = self.ik(candidate['T_base_pregrasp_tcp'], initial, avoid_collisions=False)
            return {'candidate_id': candidate['id'], 'stage': 'pregrasp_ik', 'error_code': ik.error_code.val,
                    'ik_without_collision_check': diagnostic.error_code.val,
                    'collision_check_disabled_for_diagnostic_only': True}
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name, motion.pipeline_id = 'arm', 'ompl'
        motion.planner_id = 'RRTConnectkConfigDefault'
        motion.start_state = initial
        motion.allowed_planning_time = 2.0
        motion.num_planning_attempts = 2
        motion.max_velocity_scaling_factor = motion.max_acceleration_scaling_factor = 0.1
        constraints = Constraints()
        targets = dict(zip(ik.solution.joint_state.name, ik.solution.joint_state.position))
        for i in range(1, 7):
            joint = JointConstraint()
            joint.joint_name = f'joint{i}'
            joint.position, joint.tolerance_above, joint.tolerance_below, joint.weight = targets[joint.joint_name], 0.005, 0.005, 1.0
            constraints.joint_constraints.append(joint)
        motion.goal_constraints = [constraints]
        response = self.service(GetMotionPlan, '/plan_kinematic_path', request)
        plan = response.motion_plan_response
        if plan.error_code.val != 1 or not plan.trajectory.joint_trajectory.points:
            return {'candidate_id': candidate['id'], 'stage': 'pregrasp_path', 'error_code': plan.error_code.val}
        approach = GetCartesianPath.Request()
        approach.header.frame_id = 'base_link'
        approach.group_name, approach.link_name = 'arm', 'tcp_link'
        approach.start_state = deepcopy(initial)
        positions = dict(zip(plan.trajectory.joint_trajectory.joint_names,
                             plan.trajectory.joint_trajectory.points[-1].positions))
        approach.start_state.joint_state.position = [
            positions.get(name, value) for name, value in
            zip(initial.joint_state.name, initial.joint_state.position)]
        approach.waypoints = [pose_from_matrix(candidate['T_base_tcp'])]
        approach.max_step, approach.revolute_jump_threshold = 0.005, 0.15
        approach.prismatic_jump_threshold = 0.01
        approach.avoid_collisions = True
        approach.max_velocity_scaling_factor = approach.max_acceleration_scaling_factor = 0.1
        cartesian = self.service(GetCartesianPath, '/compute_cartesian_path', approach)
        outcome = {'candidate_id': candidate['id'], 'stage': 'cartesian_approach',
                   'error_code': cartesian.error_code.val, 'approach_fraction': cartesian.fraction}
        if cartesian.error_code.val == 1 and cartesian.fraction >= 0.999:
            outcome['trajectories'] = [plan.trajectory, cartesian.solution]
        return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-spec', type=Path, help='Measured upright cylinder specification JSON')
    parser.add_argument('--plan', action='store_true', help='Request plans via planning services only')
    parser.add_argument('--clear-scene', action='store_true', help='Remove only this preview node\'s two collision objects')
    parser.add_argument('--keep-alive', action='store_true', help='Keep the latched RViz markers available')
    parser.add_argument('--out', type=Path, default=Path('records/cylinder_real/grasp_preview.json'))
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = CylinderPreview()
    if args.clear_scene:
        try:
            node.update_scene()
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return 0
    report = {'execution_ready': False, 'plan_only': True,
              'scene_scope': 'local tabletop patch and observed red band; unseen geometry unmodelled',
              'tcp_scope': 'official model fingertip midpoint, not an independently measured contact TCP'}
    try:
        node.wait(node.fresh, 20)
        sample = deepcopy(node.observation)
        report['observation'] = sample
        if args.object_spec:
            spec = json.loads(args.object_spec.read_text())
            if not spec.get('use_for_collision_planning'):
                raise ValueError('Object specification is not enabled for collision planning')
            sample['base']['full_geometry'] = full_cylinder_geometry(sample['base'], spec)
            report['scene_scope'] = 'measured full cylinder and local tabletop patch; other obstacles unmodelled'
        initial = RobotState()
        initial.joint_state = deepcopy(node.joints)
        initial.is_diff = False
        report['initial_real_joint_state'] = {
            'names': list(initial.joint_state.name),
            'positions': list(initial.joint_state.position)}
        node.wait(lambda: node.buffer.can_transform('link6', 'tcp_link', Time()), 5)
        tcp = transform_from_ros(node.buffer.lookup_transform('link6', 'tcp_link', Time()))
        model = installed_gripper_model_path()
        tip = model_tip_transform(model)
        report.update(observation=sample, T_link6_tip=tip.tolist(), T_link6_tcp=tcp.tolist())
        candidates = grasp_candidates(sample['base'], tip, tcp, read_gripper_limit(model)[1])
        report['candidates'] = candidates
        node.markers(sample['base'], candidates)
        report['status'] = 'candidates_displayed'
        if args.plan:
            if sample['base'].get('support_patch') is None:
                raise ValueError('No valid support plane; collision-aware preview planning is blocked')
            node.update_scene(sample['base'])
            # This is a hypothetical robot state, not a command to the gripper.
            if 'gripper' not in initial.joint_state.name:
                raise ValueError('No actual gripper joint feedback')
            index = initial.joint_state.name.index('gripper')
            actual_opening = initial.joint_state.position[index]
            initial.joint_state.position[index] = candidates[0]['opening_m']
            report.update(actual_gripper_width_m=actual_opening,
                          assumed_gripper_width_m=candidates[0]['opening_m'], attempts=[])
            selected = None
            for candidate in candidates:
                node.check_snapshot(sample, initial)
                print(f'Checking candidate {candidate["id"]} ({candidate["kind"]})', flush=True)
                attempt = node.plan_candidate(candidate, initial)
                trajectories = attempt.pop('trajectories', None)
                report['attempts'].append(attempt)
                if trajectories is not None:
                    node.check_snapshot(sample, initial)
                    selected = candidate['id']
                    path = DisplayTrajectory()
                    path.model_id, path.trajectory_start, path.trajectory = 'agx_arm', initial, trajectories
                    # Visual display only; no execute action or controller topic is used.
                    node.path_pub.publish(path)
                    report['trajectory_point_counts'] = [len(t.joint_trajectory.points) for t in trajectories]
                    report['trajectories'] = [{
                        'joint_names': list(t.joint_trajectory.joint_names),
                        'points': [{'positions': list(p.positions), 'velocities': list(p.velocities),
                                    'accelerations': list(p.accelerations),
                                    'time_from_start_s': p.time_from_start.sec + p.time_from_start.nanosec * 1e-9}
                                   for p in t.joint_trajectory.points]}
                        for t in trajectories]
                    break
            report['selected_candidate_id'] = selected
            report['status'] = 'preview_path_found' if selected is not None else 'all_candidates_rejected'
            node.markers(sample['base'], candidates, selected)
        report['requires_fresh_observation_before_execution'] = True
    except (ValueError, RuntimeError, TimeoutError, TransformException) as exc:
        report.update(status='blocked', reason=str(exc))
        print(str(exc), file=sys.stderr)
    except (KeyboardInterrupt, ExternalShutdownException):
        report.update(status='interrupted')
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(f'Saved {args.out}: {report.get("status", "cleared")}', flush=True)
    try:
        if args.keep_alive and report.get('status') != 'interrupted':
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if report.get('status') in ('preview_path_found', 'candidates_displayed') else 1


if __name__ == '__main__':
    sys.exit(main())
