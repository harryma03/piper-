#!/usr/bin/env python3
"""Plan small wrist observation motions; execute only with --execute.

Requires the real camera, unique real driver, ros2_control and MoveIt.
Leaves gripper untouched. Any failure stops the sequence; no blind return.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.time import Time
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState
from moveit_msgs.srv import GetMotionPlan
from std_srvs.srv import SetBool
from piperx_lab.cylinder_preview import CylinderPreview
from piperx_lab.cylinder_grasp import support_patch
from piperx_lab.handeye import transform_from_ros
from piperx_lab.transforms import load_handeye
from piperx_lab.model import read_joint_limits, installed_urdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('records/cylinder_real/multiview'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'execute_requested': args.execute, 'moves': [], 'status': 'starting'}
    rclpy.init()
    node = CylinderPreview()
    action = ActionClient(node, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
    names = [f'joint{i}' for i in range(1, 7)]
    active = None

    def fresh_joints():
        node.wait(lambda: node.joints is not None and abs(node.age(node.joints.header.stamp)) < .3, 5)
        return deepcopy(node.joints)

    def capture(label):
        pose_before = fresh_joints()
        folder = args.out / label
        run = subprocess.run([sys.executable, str(root/'scripts/diagnose_cylinder_geometry.py'),
                              '--out', str(folder), '--known-diameter-m', '.046'],
                             capture_output=True, text=True, timeout=25)
        if run.returncode:
            raise RuntimeError('Capture failed: ' + run.stderr[-1000:])
        pose_after = fresh_joints()
        a, b = dict(zip(pose_before.name, pose_before.position)), dict(zip(pose_after.name, pose_after.position))
        if any(abs(a[n]-b[n]) > .005 for n in names):
            raise ValueError('Arm was not stationary during capture')
        node.wait(lambda: node.buffer.can_transform('base_link', 'link6', Time()) and
                  abs(node.age(node.buffer.lookup_transform('base_link', 'link6', Time()).header.stamp)) < .3, 5)
        tf = node.buffer.lookup_transform('base_link', 'link6', Time())
        if abs(node.age(tf.header.stamp)) > .3:
            raise ValueError('Stale wrist TF')
        handeye = load_handeye(root/'calibration/handeye_result.json')
        if not handeye['quality_pass']:
            raise ValueError('Hand-eye calibration quality check failed')
        transform = transform_from_ros(tf) @ handeye['T_wrist_camera']
        (folder/'robot_state.json').write_text(json.dumps({'joint_names':list(pose_after.name),
            'positions':list(pose_after.position),'T_base_camera':transform.tolist(),
            'scope':'stationary capture with post-capture fresh TF; not moving-frame synchronization'},indent=2)+'\n')
        return folder, transform

    def plan(target):
        if node.count_publishers('/feedback/joint_states') != 1:
            raise ValueError('Expected exactly one real feedback publisher')
        state = fresh_joints()
        current = dict(zip(state.name, state.position))
        limits = read_joint_limits(installed_urdf_path())
        if any(not limits[n][0] <= target[n] <= limits[n][1] or abs(target[n]-current[n]) > .075 for n in names):
            raise ValueError('View goal exceeds model limit or 0.075 rad movement bound')
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'arm'
        req.motion_plan_request.start_state = RobotState(joint_state=state, is_diff=False)
        req.motion_plan_request.allowed_planning_time = 3.
        req.motion_plan_request.max_velocity_scaling_factor = .03
        req.motion_plan_request.max_acceleration_scaling_factor = .03
        req.motion_plan_request.planner_id = 'RRTConnectkConfigDefault'
        c = Constraints()
        for n in names:
            c.joint_constraints.append(JointConstraint(joint_name=n, position=target[n], tolerance_above=.001,tolerance_below=.001,weight=1.))
        req.motion_plan_request.goal_constraints = [c]
        result = node.service(GetMotionPlan, '/plan_kinematic_path', req).motion_plan_response
        if result.error_code.val != 1 or not result.trajectory.joint_trajectory.points:
            raise ValueError(f'Collision-aware plan rejected: {result.error_code.val}')
        trajectory = result.trajectory.joint_trajectory
        start = np.array([current[n] for n in trajectory.joint_names])
        for point in trajectory.points:
            if not np.all(np.isfinite(point.positions)) or np.max(abs(np.array(point.positions)-start)) > .08:
                raise ValueError('Planned path leaves small local joint envelope')
        duration = trajectory.points[-1].time_from_start.sec + trajectory.points[-1].time_from_start.nanosec*1e-9
        scale = max(1., 5./max(duration,.001))
        for point in trajectory.points:
            t = round((point.time_from_start.sec + point.time_from_start.nanosec*1e-9)*scale*1e9)
            point.time_from_start.sec, point.time_from_start.nanosec = divmod(t,10**9)
            point.velocities = [v/scale for v in point.velocities]
            point.accelerations = [v/scale**2 for v in point.accelerations]
        return trajectory, current

    try:
        node.wait(lambda: node.count_publishers('/feedback/joint_states') == 1, 5)
        baseline, transform = capture('baseline')
        cloud = np.load(baseline/'frame.npz')
        body = cloud['xyz'][cloud['mask']] @ transform[:3,:3].T + transform[:3,3]
        reference = np.median(body,axis=0)
        sparse = cloud['xyz'][::5,::5].reshape(-1,3)
        sparse = sparse[sparse[:,2]>0] @ transform[:3,:3].T + transform[:3,3]
        patch = support_patch(sparse,reference)
        normal = np.array(patch['normal'])
        scene = {'support_patch':patch, 'full_geometry':{
            'height_m':.18, 'diameter_m':.14,
            'center_xyz_m':(np.array(patch['surface_point_m'])+.04*normal).tolist(),
            'axis_direction':normal.tolist()}}
        node.update_scene(scene)
        report['collision_scene'] = 'local observed table and conservative target envelope; other obstacles unmodelled'
        report['scene'] = scene
        q = fresh_joints();base = {n:dict(zip(q.name,q.position))[n] for n in names}
        report['start'] = base
        for label,joint,offset in [('wrist4_plus','joint4',.07),('return1',None,0.),('wrist5_minus','joint5',-.07),('return2',None,0.)]:
            target=base.copy()
            if joint:target[joint]+=offset
            path, initial = plan(target)
            entry={'label':label,'target':target,'planned_points':len(path.points),'executed':False,
                   'trajectory': {'joint_names':list(path.joint_names), 'points':[
                       {'positions':list(p.positions), 'time_from_start_s':p.time_from_start.sec+p.time_from_start.nanosec*1e-9}
                       for p in path.points]}}
            report['moves'].append(entry)
            print(label,entry,flush=True)
            if not args.execute:
                # Do not plan a return from a hypothetical state as a live path.
                if joint is None:continue
                continue
            current=dict(zip(fresh_joints().name,node.joints.position))
            if any(abs(current[n]-initial[n])>.005 for n in names):raise ValueError('Start changed after planning')
            if not action.wait_for_server(timeout_sec=3):raise RuntimeError('No arm trajectory action')
            response=node.service(SetBool,'/enable_agx_arm',SetBool.Request(data=True))
            if not response.success:raise RuntimeError(response.message)
            goal=FollowJointTrajectory.Goal(trajectory=path)
            future=action.send_goal_async(goal);node.wait(future.done,5);active=future.result()
            if not active.accepted:raise RuntimeError('Trajectory rejected')
            result=active.get_result_async();node.wait(result.done,20)
            outcome=result.result();active=None
            if outcome.status!=4 or outcome.result.error_code!=0:raise RuntimeError('Trajectory execution failed')
            node.wait(lambda: node.joints is not None and abs(node.age(node.joints.header.stamp))<.3 and
                all(abs(dict(zip(node.joints.name,node.joints.position))[n]-target[n])<.01 for n in names),5)
            entry['executed']=True
            if joint:capture(label)
        report['status']='completed' if args.execute else 'planning_only'
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='stopped',reason=str(exc));print(str(exc),file=sys.stderr)
        if active is not None and active.accepted:
            cancel=active.cancel_goal_async()
            try:node.wait(cancel.done,3)
            except Exception:pass
    finally:
        if args.execute:
            try:node.service(SetBool,'/control_enable',SetBool.Request(data=False))
            except Exception:pass
        (args.out/'run.json').write_text(json.dumps(report,indent=2)+'\n')
        node.destroy_node();rclpy.shutdown()
    return 0 if report['status'] in ('completed','planning_only') else 1


if __name__=='__main__':raise SystemExit(main())
