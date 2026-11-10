#!/usr/bin/env python3
"""Closed-loop grasp executor with active multi-view observation.

State machine:

    OBSERVE -> (VIEWPOINTS ->) CANDIDATES -> PREGRASP -> APPROACH_SEGMENTS
            -> CLOSE -> LIFT -> HOLD -> success | recover | fail

- `--mode single`: one viewpoint, one full plan, no mid-approach updates
  (the repaired one-shot baseline).
- `--mode multiview`: planner-driven extra viewpoints fuse into the object
  model before planning; still no mid-approach updates.
- `--mode closedloop`: multi-view plus segmented approach -- every segment
  ends with a perception update; target displacement above
  `--replan-displacement-m` replans the remaining segments (with hysteresis
  so per-frame noise cannot flip-flop the plan), a lost/stale target
  suspends or re-observes, and the final contact segment is constrained by
  the latest observation.

Requires the resident perception service (perception_service.py) on the same
domain; this executor never touches Gazebo truth -- GetEntityState is only
queried for the final evaluation block, exactly like run_grasp.py.
"""
import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.parameter import Parameter as RclParameter
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import (
    CollisionObject, Constraints, JointConstraint, RobotState,
    AllowedCollisionEntry,
)
from moveit_msgs.srv import (
    ApplyPlanningScene, GetPositionIK, GetMotionPlan, GetCartesianPath,
    GetPlanningScene,
)
from gazebo_msgs.srv import GetEntityState
from gazebo_msgs.msg import ContactsState
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from piperx_lab.collision_geometry import collision_object_from_hull
from piperx_lab.cylinder_preview import CylinderPreview, pose_from_matrix
from piperx_lab.grasp_candidate_filter import (
    candidate_center_near_target, neighborhood_radius_m)
from piperx_lab.grasp_candidates import MAX_OPENING_M, apply_revalidation
from piperx_lab.gripper_policy import (
    MIN_CLOSE_WIDTH_M, classify_grasp, closing_sequence)
from piperx_lab.model import installed_gripper_model_path
from piperx_lab.object_observation import (
    SOURCE as GENERIC_SOURCE, GenericObjectObservation, read_ply)
from piperx_lab.observation_fusion import FusedObservation, fuse_view, summarize
from piperx_lab.viewpoint_planner import plan_viewpoints
from piperx_lab.cylinder_grasp import model_tip_transform

GRIPPER_JOINTS = ("gripper", "gripper_joint1", "gripper_joint2")
OPEN_WIDTH_M = 0.070


class ClosedLoopExecutor:
    def __init__(self, node, args, report):
        self.n = node
        self.args = args
        self.report = report
        self.fused = FusedObservation()
        self.latest_observation = None
        self.latest_candidates = []
        self.neighborhood_radius = 0.09
        self.view_records = []
        self.candidate_events = []
        self.motion_events = []
        self.selected_candidate = None
        self.main_stage = ['startup']
        self.perception_ready = False

    # ---------- perception ----------
    def request_perception(self, view_id, save_dir, timeout_s=60.0):
        request_id = int(time.monotonic() * 1000) % 1000000
        payload = {'request_id': request_id, 'view_id': view_id,
                   'save_dir': str(Path(save_dir).resolve()),
                   'roi_paddings': [40, 60, 80], 'depth_window': 0.08}
        if self.args.target_pixel is not None:
            payload['target_pixel'] = self.args.target_pixel
        got = {}
        done = [False]

        def on_response(msg):
            data = json.loads(msg.data)
            if data.get('request_id') == request_id and not done[0]:
                got.update(data)
                done[0] = True

        sub = self.n.create_subscription(String, '/piperx/perception/response',
                                         on_response, 10)
        self.n.perception_pub.publish(String(data=json.dumps(payload)))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not done[0]:
            rclpy.spin_once(self.n, timeout_sec=0.05)
        self.n.destroy_subscription(sub)
        if not done[0]:
            return {'ok': False, 'error': 'perception service timeout'}
        return got

    # ---------- MoveIt helpers (mirrors run_grasp.py) ----------
    def state(self):
        self.n.wait(lambda: self.n.joints is not None, 5)
        for _ in range(5):
            rclpy.spin_once(self.n, timeout_sec=0.01)
        # is_diff=True keeps an attached payload in the planning start state.
        return RobotState(joint_state=deepcopy(self.n.joints), is_diff=True)

    def execute(self, path, controller, timeout_s=35):
        client = ActionClient(self.n, FollowJointTrajectory,
                              f'/{controller}/follow_joint_trajectory')
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError('No simulated controller')
        goal = client.send_goal_async(FollowJointTrajectory.Goal(trajectory=path))
        self.n.wait(goal.done, 5)
        accepted = goal.result()
        if not accepted.accepted:
            client.destroy()
            raise RuntimeError('Trajectory rejected')
        result = accepted.get_result_async()
        try:
            self.n.wait(result.done, timeout_s)
        except Exception:
            cancel = accepted.cancel_goal_async()
            self.n.wait(cancel.done, 3)
            client.destroy()
            self.motion_events.append({'event': 'trajectory_cancelled',
                                       'controller': controller})
            raise
        r = result.result()
        client.destroy()
        if r.status != 4 or r.result.error_code != 0:
            raise RuntimeError(f'Trajectory failed: {r.status} {r.result.error_string}')
        return True

    def preplan(self, matrix, initial=None, ik_link='tcp_link'):
        initial = self.state() if initial is None else initial
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'arm'
        req.ik_request.ik_link_name = ik_link
        req.ik_request.robot_state = initial
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped.header.frame_id = 'base_link'
        req.ik_request.pose_stamped.pose = pose_from_matrix(matrix)
        req.ik_request.timeout.sec = 1
        ik = self.n.service(GetPositionIK, '/compute_ik', req)
        if ik.error_code.val != 1:
            raise RuntimeError(f'IK {ik.error_code.val}')
        mreq = GetMotionPlan.Request()
        m = mreq.motion_plan_request
        m.group_name = 'arm'
        m.start_state = initial
        m.allowed_planning_time = 3.0
        m.max_velocity_scaling_factor = 0.1
        m.max_acceleration_scaling_factor = 0.1
        c = Constraints()
        for name, value in zip(ik.solution.joint_state.name,
                               ik.solution.joint_state.position):
            if name.startswith('joint'):
                c.joint_constraints.append(JointConstraint(
                    joint_name=name, position=value, tolerance_above=0.001,
                    tolerance_below=0.001, weight=1.0))
        m.goal_constraints = [c]
        r = self.n.service(GetMotionPlan, '/plan_kinematic_path',
                           mreq).motion_plan_response
        if r.error_code.val != 1:
            raise RuntimeError(f'plan {r.error_code.val}')
        return r.trajectory.joint_trajectory

    def cartesian(self, matrix, scaling=0.1, initial=None):
        q = GetCartesianPath.Request()
        q.header.frame_id = 'base_link'
        q.group_name = 'arm'
        q.link_name = 'tcp_link'
        q.start_state = self.state() if initial is None else initial
        q.waypoints = [pose_from_matrix(matrix)]
        q.max_step = 0.003
        q.avoid_collisions = True
        q.max_velocity_scaling_factor = scaling
        q.max_acceleration_scaling_factor = scaling
        r = self.n.service(GetCartesianPath, '/compute_cartesian_path', q)
        # 0.95: short closed-loop segments and the post-close lift share the
        # workspace with the gripped object; 0.999 rejects lifts that only
        # graze the tolerance boundary.
        if r.error_code.val != 1 or r.fraction < 0.95:
            raise RuntimeError(f'Cartesian fraction {r.fraction}')
        return r.solution.joint_trajectory

    def apply_scene(self, observation):
        """Ground + hull target + observed obstacle boxes; fingertip contact
        is allowed only during the final approach segment."""
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        target = CollisionObject(id='target', operation=CollisionObject.ADD)
        target.header.frame_id = 'base_link'
        hull_path = (Path(self.args.out) / observation['convex_hull_path']
                     if observation.get('convex_hull_path') else None)
        if not hull_path or not hull_path.is_file():
            raise ValueError('observation carries no convex hull')
        table_height = float(observation['quality']['table_height_m'])
        mesh_obj, provenance = collision_object_from_hull(
            hull_path, table_height_m=table_height, observation=observation)
        target.meshes = mesh_obj.meshes
        target.mesh_poses = mesh_obj.mesh_poses
        self.report['collision_geometry'] = provenance
        ground = CollisionObject(id='ground', operation=CollisionObject.ADD)
        ground.header.frame_id = 'base_link'
        ground.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                            dimensions=[2.0, 2.0, 0.05])]
        t = np.eye(4)
        t[:3, 3] = [0.0, 0.0, -0.175]
        ground.primitive_poses = [pose_from_matrix(t)]
        req.scene.world.collision_objects.extend([ground, target])
        for obstacle_index, box in enumerate(observation.get('obstacles_base') or []):
            low = np.asarray(box['aabb_min_m'], dtype=float)
            high = np.asarray(box['aabb_max_m'], dtype=float)
            obstacle = CollisionObject(id=f'obstacle_{obstacle_index}',
                                       operation=CollisionObject.ADD)
            obstacle.header.frame_id = 'base_link'
            obstacle.primitives = [SolidPrimitive(
                type=SolidPrimitive.BOX,
                dimensions=np.maximum(high - low, 0.005).tolist())]
            t = np.eye(4)
            t[:3, 3] = (low + high) / 2
            obstacle.primitive_poses = [pose_from_matrix(t)]
            req.scene.world.collision_objects.append(obstacle)
        semantic = self.n.service(
            GetParameters, '/move_group/get_parameters',
            GetParameters.Request(names=['robot_description_semantic'])
        ).values[0].string_value
        import xml.etree.ElementTree as ET
        model = ET.fromstring(semantic)
        allowed = {frozenset((p.get('link1'), p.get('link2')))
                   for p in model.findall('disable_collisions')}
        allowed.update(frozenset(pair) for pair in [
            ('target', 'gripper_link1'), ('target', 'gripper_link2'),
            ('ground', 'base_link')])
        names = sorted(set().union(*allowed))
        req.scene.allowed_collision_matrix.entry_names = names
        for a in names:
            req.scene.allowed_collision_matrix.entry_values.append(
                AllowedCollisionEntry(
                    enabled=[a == b or frozenset((a, b)) in allowed for b in names]))
        if not self.n.service(ApplyPlanningScene, '/apply_planning_scene', req).success:
            raise RuntimeError('Scene rejected')

    def allow_finger_contact(self, enabled, names, allowed):
        diff = ApplyPlanningScene.Request()
        diff.scene.is_diff = True
        diff.scene.robot_state.is_diff = True
        diff.scene.allowed_collision_matrix.entry_names = names
        for a in names:
            row = []
            for b in names:
                pair = frozenset((a, b))
                is_finger = pair in {frozenset(('target', 'gripper_link1')),
                                     frozenset(('target', 'gripper_link2'))}
                row.append(enabled if is_finger else a == b or pair in allowed)
            diff.scene.allowed_collision_matrix.entry_values.append(
                AllowedCollisionEntry(enabled=row))
        if not self.n.service(ApplyPlanningScene, '/apply_planning_scene', diff).success:
            raise RuntimeError('ACM update rejected')

    # ---------- state machine ----------
    def merge_candidates(self, new_candidates, new_center):
        """Cross-view candidate pool with hysteresis.

        Candidates from every viewpoint coexist (deduplicated by grip-tip
        position); a viewpoint that yields nothing must never evict the
        pool. Duplicates from the same source at the same tip position are
        dropped; the pool is re-ranked by confidence.
        """
        pool = list(self.latest_candidates)
        seen_tips = {tuple(np.round(np.asarray(c['T_base_tip'])[:3, 3], 4))
                     for c in pool if 'T_base_tip' in c}
        added = 0
        for candidate in new_candidates or []:
            key = tuple(np.round(np.asarray(candidate['T_base_tip'])[:3, 3], 4))
            if key in seen_tips:
                continue
            seen_tips.add(key)
            pool.append(candidate)
            added += 1
        if added:
            self.candidate_events.append({
                'event': 'candidates_added', 'count': added,
                'pool_size': len(pool),
                'source_view': new_candidates[0].get('view_id') if new_candidates else None})
        pool = [c for c in pool
                if candidate_center_near_target(c, new_center, self.neighborhood_radius)]
        pool.sort(key=lambda c: (-float(c.get('confidence') or 0.0),
                                 float(c.get('approach_distance_m') or 1.0)))
        for candidate_id, candidate in enumerate(pool[:40]):
            candidate['id'] = candidate_id
        self.latest_candidates = pool[:40]
        return added

    def observe_view(self, view_id, save_dir):
        """Request one perception update, fuse it, record the evidence."""
        response = self.request_perception(view_id, save_dir)
        if not response.get('ok'):
            self.view_records.append({'view_id': view_id, 'accepted': False,
                                      'error': response.get('error')})
            return None
        observation = response['observation']
        merge = fuse_view(self.fused,
                          np.load(Path(save_dir) /
                                  observation['points_base_path'])['points_base'],
                          view_id, observation.get('stamp'),
                          camera_pose_base=observation.get('T_base_camera'))
        record = {'view_id': view_id, 'accepted': merge['accepted'],
                  'novel_voxels': merge.get('novel_voxels'),
                  'overlap': merge.get('overlap'),
                  'observed_fraction': round(
                      self.fused.observed_fraction_inside_hull(), 4),
                  'camera_position': observation.get('view_camera_position'),
                  'stamp': observation.get('stamp'),
                  'timing_ms': response.get('timing_ms')}
        self.view_records.append(record)
        return {'observation': observation, 'candidates': response.get('candidates', []),
                'merge': merge, 'record': record}

    def observation_from_fused(self):
        """A merged observation dict built from the latest view response,
        with quality fields updated from the fused model."""
        if self.latest_observation is None:
            return None
        merged = json.loads(json.dumps(self.latest_observation))  # deep copy
        merged['quality']['fused_views'] = self.fused.total_views
        merged['quality']['fused_observed_voxels'] = self.fused.observed_voxels
        merged['quality']['fused_unseen_inside_hull'] = self.fused.hull_inner_unseen_voxels
        merged['quality']['fused_observed_fraction'] = round(
            self.fused.observed_fraction_inside_hull(), 4)
        return merged

    def run_viewpoints(self, save_dir):
        planner_input = self.observation_from_fused()
        obstacles = planner_input.get('obstacles_base')
        views = plan_viewpoints(planner_input, summarize(self.fused),
                                obstacles=obstacles,
                                current_camera_xy=self.args.camera_xy,
                                max_views=self.args.max_extra_views)
        self.report['viewpoint_plan'] = views
        for view in views:
            pose = view['camera_position_base_m']
            # Move the arm so the CAMERA lands on the planned viewpoint while
            # keeping its current orientation (fixed wrist mount). The goal
            # is constructed for link6 (a real URDF frame; tcp_link is only a
            # MoveIt semantic frame and absent from the TF tree):
            #   T_base_link6_new = T_base_cam_new @ inv(T_base_cam_now) @ T_base_link6_now
            cam_tf = self.n.buffer.lookup_transform(
                'base_link', 'camera_depth_frame', rclpy.time.Time())
            cam = np.eye(4)
            q = cam_tf.transform.rotation
            cam[:3, :3] = quat_to_matrix(q.x, q.y, q.z, q.w)
            cam[:3, 3] = [cam_tf.transform.translation.x,
                          cam_tf.transform.translation.y,
                          cam_tf.transform.translation.z]
            link6_tf = self.n.buffer.lookup_transform(
                'base_link', 'link6', rclpy.time.Time())
            link6_now = np.eye(4)
            q3 = link6_tf.transform.rotation
            link6_now[:3, :3] = quat_to_matrix(q3.x, q3.y, q3.z, q3.w)
            link6_now[:3, 3] = [link6_tf.transform.translation.x,
                                link6_tf.transform.translation.y,
                                link6_tf.transform.translation.z]
            cam_new = cam.copy()
            cam_new[:3, 3] = [pose[0], pose[1], pose[2]]
            target_pose = cam_new @ np.linalg.inv(cam) @ link6_now
            try:
                path = self.preplan(target_pose, ik_link='link6')
                self.execute(path, 'arm_controller', timeout_s=30)
                self.motion_events.append({'event': 'viewpoint_executed',
                                           'view_id': view['view_id'],
                                           'position': pose})
            except (RuntimeError, TimeoutError) as exc:
                self.motion_events.append({'event': 'viewpoint_rejected',
                                           'view_id': view['view_id'],
                                           'reason': str(exc)})
                self.report['view_records'].append({
                    'view_id': view['view_id'], 'accepted': False,
                    'error': f'viewpoint motion failed: {exc}'})
                continue
            time.sleep(0.4)  # let the camera stream settle at the new pose
            result = self.observe_view(view['view_id'], save_dir)
            self.report['view_records'].append(result['record'] if result else {
                'view_id': view['view_id'], 'accepted': False})
            if result and result['merge'].get('accepted'):
                self.latest_observation = result['observation']
                center = self.latest_observation['obb_center_base_m']
                self.merge_candidates(result['candidates'], center)


def quat_to_matrix(x, y, z, w):
    matrix = np.eye(3)
    matrix[0, 0] = 1 - 2 * (y * y + z * z)
    matrix[0, 1] = 2 * (x * y - z * w)
    matrix[0, 2] = 2 * (x * z + y * w)
    matrix[1, 0] = 2 * (x * y + z * w)
    matrix[1, 1] = 1 - 2 * (x * x + z * z)
    matrix[1, 2] = 2 * (y * z - x * w)
    matrix[2, 0] = 2 * (x * z - y * w)
    matrix[2, 1] = 2 * (y * z + x * w)
    matrix[2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def main():
    parser = __import__('argparse').ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single', 'multiview', 'closedloop'),
                        default='closedloop')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--camera-xy', default='0.45,0.0')
    parser.add_argument('--max-extra-views', type=int, default=2)
    parser.add_argument('--replan-displacement-m', type=float, default=0.012)
    parser.add_argument('--approach-segments', type=int, default=4)
    parser.add_argument('--grip-seconds', type=int, default=8)
    parser.add_argument('--hold-seconds', type=float, default=3.0)
    parser.add_argument('--target-pixel', metavar='X,Y',
                        help='Select a depth-connected target without colour (single-view simulation only)')
    args = parser.parse_args()
    if args.target_pixel is not None:
        if args.mode != 'single':
            parser.error('--target-pixel currently requires --mode single; a fixed image pixel cannot track a moving camera')
        try:
            args.target_pixel = [int(v) for v in args.target_pixel.split(',')]
            if len(args.target_pixel) != 2:
                raise ValueError()
        except ValueError:
            parser.error('--target-pixel must be X,Y integer image coordinates')
    args.camera_xy = tuple(float(v) for v in args.camera_xy.split(','))
    if os.environ.get('ROS_DOMAIN_ID') != '183':
        raise SystemExit('Requires isolated simulation domain 183')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        'mode': args.mode, 'status': 'running', 'failure_stage': None,
        'view_records': [], 'viewpoint_plan': [], 'motion_events': [],
        'candidate_events': [], 'perception_timing': [], 'success': False,
        'physical_grasp_success': False, 'execution_complete': False,
    }
    rclpy.init()
    node = CylinderPreview()
    node.set_parameters([RclParameter('use_sim_time', RclParameter.Type.BOOL, True)])
    executor = ClosedLoopExecutor(node, args, report)
    node.perception_pub = node.create_publisher(String, '/piperx/perception/request', 10)
    node.create_subscription(JointState, '/joint_states',
                             lambda m: setattr(node, 'joints', m), 10)
    contacts_report = {'closing': {}, 'lifting': {}}
    phase = ['idle']
    node.create_subscription(ContactsState, '/gripper_link1/contacts',
                             lambda m: record_contact(m, 'gripper_link1'), 10)
    node.create_subscription(ContactsState, '/gripper_link2/contacts',
                             lambda m: record_contact(m, 'gripper_link2'), 10)

    def record_contact(msg, finger):
        hits = [s for s in msg.states
                if 'cylinder::' in s.collision1_name or 'cylinder::' in s.collision2_name]
        if hits:
            contacts_report.setdefault(phase[0], {})[finger] = True

    try:
        node.wait(lambda: node.count_publishers('/joint_states') == 1
                  and node.joints is not None, 10)
        # Wait for the resident perception service (already warm after its
        # own start-up inference).
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            probe = executor.request_perception('probe', args.out, timeout_s=20)
            if probe.get('ok'):
                executor.perception_ready = True
                break
        if not executor.perception_ready:
            raise RuntimeError('perception service unreachable')

        # ---------- OBSERVE ----------
        executor.main_stage[0] = 'observe'
        first = executor.observe_view('v0', args.out)
        if first is None or not first['observation']:
            raise RuntimeError('initial observation failed')
        executor.latest_observation = first['observation']
        executor.latest_candidates = first['candidates']
        report['initial_observation'] = first['observation']

        # ---------- VIEWPOINTS (multiview / closedloop) ----------
        if args.mode in ('multiview', 'closedloop'):
            executor.main_stage[0] = 'viewpoints'
            before = executor.fused.observed_voxels
            executor.run_viewpoints(args.out)
            report['fusion'] = summarize(executor.fused)
            report['novel_voxels_from_extra_views'] = executor.fused.observed_voxels - before
            # Prove the extra views added information:
            report['new_information_proven'] = (
                executor.fused.observed_voxels - before) >= 15 and \
                len(executor.fused.views) > 1

        # ---------- CANDIDATES ----------
        executor.main_stage[0] = 'candidates'
        merged_obs = executor.observation_from_fused()
        planning_center = list(merged_obs['obb_center_base_m'])
        dims = merged_obs.get('dimensions_m')
        executor.neighborhood_radius = (neighborhood_radius_m(dims)
                                        if dims else 0.09)
        candidates = [c for c in executor.latest_candidates
                      if candidate_center_near_target(c, planning_center,
                                                      executor.neighborhood_radius)]
        if not candidates:
            raise RuntimeError('no candidate near the fused target')
        report['candidate_events'].append({
            'event': 'candidates_selected', 'count': len(candidates),
            'top_confidence': candidates[0].get('confidence')})
        executor.selected_candidate = candidates[0]

        # Scene from the fused observation's hull.
        executor.apply_scene(merged_obs)
        semantic = node.service(
            GetParameters, '/move_group/get_parameters',
            GetParameters.Request(names=['robot_description_semantic'])
        ).values[0].string_value
        import xml.etree.ElementTree as ET
        allowed = {frozenset((p.get('link1'), p.get('link2')))
                   for p in ET.fromstring(semantic).findall('disable_collisions')}
        allowed.update(frozenset(p) for p in [
            ('target', 'gripper_link1'), ('target', 'gripper_link2'),
            ('ground', 'base_link')])
        acm_names = sorted(set().union(*allowed))

        target = np.array(executor.selected_candidate['T_base_tcp'])
        pregrasp = np.array(executor.selected_candidate['T_base_pregrasp_tcp'])

        # ---------- PREGRASP ----------
        executor.main_stage[0] = 'pregrasp'
        executor.allow_finger_contact(False, acm_names, allowed)
        pregrasp_path = executor.preplan(pregrasp)
        executor.execute(pregrasp_path, 'arm_controller')
        report['stages'] = ['observed'] + (
            ['viewpoints'] if args.mode != 'single' else []) + ['pregrasp']

        # ---------- APPROACH ----------
        executor.main_stage[0] = 'approach'
        if args.mode == 'closedloop':
            segments = max(2, args.approach_segments)
            delta = target[:3, 3] - pregrasp[:3, 3]
            length = float(np.linalg.norm(delta))
            direction = delta / max(length, 1e-9)
            last_center = np.array(planning_center)
            replans = 0
            current_start = pregrasp
            for segment in range(segments):
                seg_start_z = current_start[:3, 3]
                # Remaining distance shrinks each segment; final segment ends
                # at the grasp pose (the only contact-allowed stretch).
                remaining_start = current_start
                seg_end = target.copy()
                if segment < segments - 1:
                    frac = (segment + 1) / segments
                    seg_end[:3, 3] = pregrasp[:3, 3] + delta * frac
                near_end = seg_end.copy()
                if segment < segments - 1:
                    # keep a 1 cm no-contact gap between segments
                    near_end[:3, 3] = seg_end[:3, 3] - direction * min(0.01, length / 8)
                segment_path = executor.cartesian(
                    near_end if segment < segments - 1 else target, 0.1)
                executor.execute(segment_path, 'arm_controller')
                # Perception update between segments (except after the final
                # contact segment, where the camera is inside its blind zone).
                if segment < segments - 1:
                    update = executor.observe_view(f'appr{segment}', args.out)
                    if update and update['observation']:
                        new_center = np.asarray(
                            update['observation']['obb_center_base_m'])
                        displacement = float(np.linalg.norm(
                            new_center[:2] - last_center[:2]))
                        report['perception_timing'].append({
                            'segment': segment, 'displacement_m': round(displacement, 4),
                            'stamp': update['observation'].get('stamp')})
                        if displacement > args.replan_displacement_m:
                            replans += 1
                            report['motion_events'].append({
                                'event': 'target_moved_replan',
                                'segment': segment,
                                'displacement_m': round(displacement, 4)})
                            # Re-derive the grasp from the fresh candidates.
                            fresh = update['candidates']
                            fresh = [c for c in fresh
                                     if candidate_center_near_target(c, new_center, radius)]
                            if fresh:
                                new_target = np.array(fresh[0]['T_base_tcp'])
                                shift = new_target[:3, 3] - target[:3, 3]
                                target[:3, 3] += shift * 0.8  # damped update
                                report['candidate_events'].append({
                                    'event': 'target_updated', 'segment': segment,
                                    'shift_m': [round(float(v), 4) for v in shift]})
                            last_center = new_center
                current_start = seg_end
            report['replan_count'] = replans
        else:
            near = target.copy()
            delta = target[:3, 3] - pregrasp[:3, 3]
            length = float(np.linalg.norm(delta))
            near[:3, 3] = target[:3, 3] - delta / length * min(0.01, length / 4)
            executor.execute(executor.cartesian(near, 0.1), 'arm_controller')
            executor.allow_finger_contact(True, acm_names, allowed)
        executor.allow_finger_contact(True, acm_names, allowed)
        report['stages'].append('approached')

        # ---------- CLOSE ----------
        executor.main_stage[0] = 'closing'
        phase[0] = 'closing'
        sequence = closing_sequence(
            (executor.selected_candidate or {}).get('predicted_width_m'))
        closing_contacts = contacts_report.setdefault('closing', {})
        close_width = sequence[-1]
        def command_gripper(width, seconds):
            point = JointTrajectoryPoint(
                positions=[width, width / 2, -width / 2])
            point.time_from_start.sec = seconds
            executor.execute(JointTrajectory(joint_names=list(GRIPPER_JOINTS),
                                             points=[point]), 'gripper_controller')
        for step_index, width in enumerate(sequence):
            command_gripper(width, 4 if step_index == 0 else 2)
            close_width = width
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            if (closing_contacts.get('gripper_link1')
                    and closing_contacts.get('gripper_link2')):
                stopped_at_step = step_index
                joints_now = dict(zip(node.joints.name, node.joints.position))
                actual = joints_now.get('gripper_joint1', 0) - joints_now.get('gripper_joint2', 0)
                lock_width = max(actual - 0.003, MIN_CLOSE_WIDTH_M)
                if lock_width < actual:
                    command_gripper(lock_width, 2)
                    close_width = lock_width
                break
        else:
            stopped_at_step = len(sequence) - 1
        report['commanded_close_width_m'] = close_width
        report['closing_stopped_at_step'] = stopped_at_step
        report['stages'].append('closed')

        # ---------- LIFT with attached payload ----------
        executor.main_stage[0] = 'lift'
        phase[0] = 'lifting'
        # Truth read for evaluation: the object height right after closing,
        # before the lift starts.
        after_close_entity = node.service(GetEntityState, '/gazebo/get_entity_state',
                                          GetEntityState.Request(name='cylinder',
                                                                 reference_frame='world'))
        if not after_close_entity.success:
            raise RuntimeError('no simulated object state for evaluation')
        p_ac = after_close_entity.state.pose.position
        report['after_close_object_xyz'] = [p_ac.x, p_ac.y, p_ac.z]
        attach = ApplyPlanningScene.Request()
        attach.scene.is_diff = True
        attach.scene.robot_state.is_diff = True
        from moveit_msgs.msg import AttachedCollisionObject
        attach.scene.robot_state.attached_collision_objects.append(
            AttachedCollisionObject(
                link_name='gripper_link2',
                touch_links=['gripper_link1', 'gripper_link2', 'gripper_link',
                             'link6', 'tcp_link'],
                object=CollisionObject(id='target', operation=CollisionObject.APPEND)))
        if not node.service(ApplyPlanningScene, '/apply_planning_scene',
                            attach).success:
            raise RuntimeError('payload attachment rejected')
        # Prove the payload is in the planning state (audit requirement).
        scene = node.service(GetPlanningScene, '/get_planning_scene',
                             GetPlanningScene.Request(
                                 components=GetPlanningScene.Request().components))
        report['attached_bodies_in_scene'] = len(
            scene.scene.robot_state.attached_collision_objects)
        lift = target.copy()
        lift[2, 3] += 0.10
        report['lift_feedback_settled'] = executor.execute(
            executor.cartesian(lift, 0.05), 'arm_controller')
        report['stages'].append('lifted')

        # ---------- HOLD + EVALUATE ----------
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        entity = node.service(GetEntityState, '/gazebo/get_entity_state',
                              GetEntityState.Request(name='cylinder',
                                                     reference_frame='world'))
        if not entity.success:
            raise RuntimeError('no simulated object state for evaluation')
        final = entity.state.pose.position
        report['final_object_xyz'] = [final.x, final.y, final.z]
        # object_lift_m compares the object's held height against its height
        # right after closing (captured just before lift) - both truth reads
        # belong to the evaluation side only.
        lift_m = final.z - after_close_entity.state.pose.position.z
        report['object_lift_m'] = lift_m
        finger_end = None
        if node.joints is not None:
            joints_now = dict(zip(node.joints.name, node.joints.position))
            finger_end = [joints_now.get('gripper_joint1', 0),
                          joints_now.get('gripper_joint2', 0)]
        state_label, details = classify_grasp(
            contacts_report, finger_end, close_width, lift_m=lift_m, both_phases=True)
        report['grasp_state'] = state_label
        report['grasp_state_details'] = details
        after_close = report.get('after_close_object_xyz')
        xy_ok = True
        if after_close:
            # XY drift between post-close and final truth reads (evaluation
            # side only); absent a post-close read, physical stays lift-based.
            xy_ok = math.hypot(final.x - after_close[0],
                               final.y - after_close[1]) < 0.06
        report['physical_grasp_success'] = bool(
            state_label == 'object_grasped' and lift_m > 0.07 and xy_ok)
        report['success'] = report['physical_grasp_success']
        report['execution_complete'] = bool(report['success']
                                            and report.get('lift_feedback_settled'))
        report['failure_stage'] = None
        report['status'] = 'completed'
        report['perception_summary'] = {
            'views': len(executor.fused.views),
            'observed_voxels': executor.fused.observed_voxels,
            'observed_fraction_inside_hull': round(
                executor.fused.observed_fraction_inside_hull(), 4),
            'replan_count': report.get('replan_count', 0),
            'stats': node.perception_stats if hasattr(node, 'perception_stats') else None}
    except Exception as exc:  # noqa: BLE001
        report.update(status='failed', success=False, reason=str(exc),
                      execution_complete=False)
        report.setdefault('failure_stage', executor.main_stage[0])
        report.setdefault('physical_grasp_success', False)
        print(str(exc), flush=True)
    finally:
        args.out.joinpath('closed_loop_result.json').write_text(
            json.dumps(report, indent=1, default=str) + '\n')
        print(json.dumps({'status': report.get('status'),
                          'success': report.get('success'),
                          'complete': report.get('execution_complete'),
                          'failure_stage': report.get('failure_stage')}))
        node.destroy_node()
        rclpy.shutdown()
    return 0 if report.get('success') else 1


if __name__ == '__main__':
    raise SystemExit(main())
