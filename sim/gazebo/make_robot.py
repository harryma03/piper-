"""Generate a Piper Gazebo model with an invisible link6-mounted camera sensor.

The arm and gripper come from the official agx_arm_description xacro. The
camera transform is simulation ground truth, not a measurement of the real
mount.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import numpy as np
from scipy.spatial.transform import Rotation
import xacro
from piperx_lab.model import ARM_JOINTS, installed_urdf_path, read_joint_limits
from piperx_lab.transforms import load_handeye


CAMERA_XYZ = '0.035 0 -0.045'
CAMERA_RPY = '3.14157 0.19916 0'


def add(parent, tag, attributes=None, text=None):
    child = ET.SubElement(parent, tag, attributes or {})
    if text is not None:
        child.text = str(text)
    return child


def add_real_camera_center(root, handeye_path: Path):
    """Draw the measured depth-frame origin as a marker fixed to link6."""
    calibration = load_handeye(handeye_path)
    if not calibration['quality_pass']:
        raise ValueError(f'Real hand-eye result failed its quality check: {handeye_path}')
    xyz = calibration['T_wrist_camera'][:3, 3]
    center = add(root, 'link', {'name': 'real_camera_center'})
    inertial = add(center, 'inertial')
    add(inertial, 'mass', {'value': '0.0001'})
    add(inertial, 'inertia', {'ixx': '0.00000001', 'ixy': '0', 'ixz': '0',
                              'iyy': '0.00000001', 'iyz': '0', 'izz': '0.00000001'})
    visual = add(center, 'visual')
    add(add(visual, 'geometry'), 'sphere', {'radius': '0.015'})
    material = add(visual, 'material', {'name': 'real_camera_center_magenta'})
    add(material, 'color', {'rgba': '1 0.05 0.65 1'})
    mount = add(root, 'joint', {'name': 'real_camera_center_mount', 'type': 'fixed'})
    add(mount, 'parent', {'link': 'link6'})
    add(mount, 'child', {'link': 'real_camera_center'})
    add(mount, 'origin', {'xyz': ' '.join(f'{value:.9f}' for value in xyz),
                          'rpy': '0 0 0'})
    appearance = add(root, 'gazebo', {'reference': 'real_camera_center'})
    add(appearance, 'material', text='Gazebo/PurpleGlow')
    gazebo = add(root, 'gazebo', {'reference': 'real_camera_center_mount'})
    add(gazebo, 'preserveFixedJoint', text='true')
    return xyz


def freeze_real_sample(root, sample_path: Path, handeye_path: Path):
    """Bake saved joint angles into fixed URDF joints for a stable Gazebo view."""
    sample = json.loads(sample_path.read_text(encoding='utf-8'))
    calibration = load_handeye(handeye_path)
    if ((sample.get('robot'), sample.get('mode'), sample.get('base_frame'),
         sample.get('wrist_frame'), sample.get('camera_frame')) !=
            ('piper', 'real', 'base_link', 'link6', calibration['camera_frame'])):
        raise ValueError('Expected a compatible real Piper sample at link6')
    limits = read_joint_limits(installed_urdf_path())
    for name in ARM_JOINTS:
        angle = float(sample['joint_positions_rad'][name])
        if not math.isfinite(angle) or not limits[name][0] <= angle <= limits[name][1]:
            raise ValueError(f'{name} is outside the Piper model limits')
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.get('type') not in ('revolute', 'continuous'):
            raise ValueError(f'Expected a revolute URDF joint: {name}')
        origin = joint.find('origin')
        axis = joint.find('axis')
        rpy = [float(value) for value in origin.get('rpy', '0 0 0').split()]
        xyz = np.array([float(value) for value in axis.get('xyz').split()])
        axis_length = np.linalg.norm(xyz)
        if axis_length == 0:
            raise ValueError(f'Empty axis for {name}')
        orientation = Rotation.from_euler('xyz', rpy) * Rotation.from_rotvec(
            angle * xyz / axis_length)
        origin.set('rpy', ' '.join(f'{value:.12f}' for value in
                                     orientation.as_euler('xyz')))
        joint.set('type', 'fixed')
        for tag in ('axis', 'limit', 'dynamics', 'safety_controller', 'calibration'):
            for element in joint.findall(tag):
                joint.remove(element)
    world_joint = root.find("joint[@name='world_to_base_link']")
    world_link = root.find("link[@name='world']")
    if world_joint is not None:
        root.remove(world_joint)
    if world_link is not None:
        root.remove(world_link)
    print(f'Fixed six Piper joints to real sample {sample_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--real-camera-center', type=Path, metavar='HANDEYE_JSON',
                        help='Show the real calibrated camera_depth_frame origin as a magenta sphere')
    parser.add_argument('--frozen-sample', type=Path, metavar='SAMPLE_JSON',
                        help='Fix the six arm joints to a real calibration sample for a stable Gazebo view')
    parser.add_argument('--rgbd', action='store_true',
                        help='Use a depth camera that provides aligned RGB and optical-Z depth')
    args = parser.parse_args()
    if args.frozen_sample and not args.real_camera_center:
        parser.error('--frozen-sample requires --real-camera-center')
    if args.frozen_sample and args.rgbd:
        parser.error('--rgbd requires the dynamic Gazebo model')
    if args.frozen_sample and args.output.suffix.lower() != '.urdf':
        parser.error('a frozen model output path must end in .urdf')
    share = Path(get_package_share_directory('agx_arm_description'))
    source = share / 'agx_arm_urdf/piper/urdf/piper_with_gripper_description.xacro'
    root = ET.fromstring(xacro.process_file(str(source)).toxml())
    prefix = 'package://agx_arm_description/'
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith(prefix):
            mesh.set('filename', (share / filename[len(prefix):]).resolve().as_uri())

    if not args.frozen_sample:
        camera = add(root, 'link', {'name': 'camera_link'})
        inertial = add(camera, 'inertial')
        add(inertial, 'mass', {'value': '0.08'})
        add(inertial, 'inertia', {'ixx': '0.00002', 'ixy': '0', 'ixz': '0',
                                  'iyy': '0.00002', 'iyz': '0', 'izz': '0.00002'})
        joint = add(root, 'joint', {'name': 'sim_camera_mount', 'type': 'fixed'})
        add(joint, 'parent', {'link': 'link6'})
        add(joint, 'child', {'link': 'camera_link'})
        add(joint, 'origin', {'xyz': CAMERA_XYZ, 'rpy': CAMERA_RPY})
        add(root, 'link', {'name': 'camera_depth_frame'})
        optical = add(root, 'joint', {'name': 'sim_camera_optical', 'type': 'fixed'})
        add(optical, 'parent', {'link': 'camera_link'})
        add(optical, 'child', {'link': 'camera_depth_frame'})
        add(optical, 'origin', {'xyz': '0 0 0', 'rpy': '-1.5707963 0 -1.5707963'})

    real_center_xyz = None
    if args.real_camera_center:
        real_center_xyz = add_real_camera_center(root, args.real_camera_center)
    if args.frozen_sample:
        freeze_real_sample(root, args.frozen_sample, args.real_camera_center)

    for link in root.findall('link'):
        gazebo = add(root, 'gazebo', {'reference': link.attrib['name']})
        add(gazebo, 'gravity', text='false')
    if not args.frozen_sample:
        preserve = add(root, 'gazebo', {'reference': 'sim_camera_mount'})
        add(preserve, 'preserveFixedJoint', text='true')

        gazebo = add(root, 'gazebo', {'reference': 'camera_link'})
        sensor = add(gazebo, 'sensor', {'name': 'handeye_color',
                                        'type': 'depth' if args.rgbd else 'camera'})
        add(sensor, 'always_on', text='true')
        add(sensor, 'update_rate', text='15')
        camera_sensor = add(sensor, 'camera', {'name': 'handeye_color'})
        add(camera_sensor, 'horizontal_fov', text='1.858776')
        image = add(camera_sensor, 'image')
        add(image, 'width', text='640')
        add(image, 'height', text='480')
        add(image, 'format', text='R8G8B8')
        if args.rgbd:
            add(camera_sensor, 'depth_camera')
        clip = add(camera_sensor, 'clip')
        add(clip, 'near', text='0.02')
        add(clip, 'far', text='3.0')
        plugin = add(sensor, 'plugin', {'name': 'handeye_camera_ros',
                                        'filename': 'libgazebo_ros_camera.so'})
        ros = add(plugin, 'ros')
        add(ros, 'namespace', text='/sim/camera')
        add(plugin, 'camera_name', text='aligned')
        add(plugin, 'frame_name', text='camera_depth_frame')
        if args.rgbd:
            add(plugin, 'min_depth', text='0.02')
            add(plugin, 'max_depth', text='3.0')

    if not args.frozen_sample:
        model_gazebo = add(root, 'gazebo')
        trajectory = add(model_gazebo, 'plugin', {
            'name': 'gazebo_ros_joint_pose_trajectory',
            'filename': 'libgazebo_ros_joint_pose_trajectory.so',
        })
        add(trajectory, 'update_rate', text='100')
        state_pub = add(model_gazebo, 'plugin', {
            'name': 'gazebo_ros_joint_state_publisher',
            'filename': 'libgazebo_ros_joint_state_publisher.so',
        })
        ros = add(state_pub, 'ros')
        add(ros, 'remapping', text='joint_states:=/control/joint_states')
        add(state_pub, 'update_rate', text='30')
        for index in range(1, 7):
            add(state_pub, 'joint_name', text=f'joint{index}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding='unicode', xml_declaration=False)
    if args.frozen_sample:
        converted = subprocess.run(['gz', 'sdf', '-p', str(args.output)],
                                   check=True, capture_output=True, text=True)
        sdf_root = ET.fromstring(converted.stdout)
        model = sdf_root.find('model')
        if model is None:
            raise ValueError('Gazebo did not generate a model from the frozen URDF')
        static = model.find('static')
        if static is None:
            static = add(model, 'static')
        static.text = 'true'
        sdf_path = args.output.with_suffix('.sdf')
        ET.ElementTree(sdf_root).write(sdf_path, encoding='unicode',
                                       xml_declaration=False)
        print(f'Saved static Gazebo model {sdf_path}')
    print(f'Saved {args.output}')
    if not args.frozen_sample:
        print(f'Invisible simulated {"RGB-D" if args.rgbd else "color"} sensor: '
              f'T_link6_camera_link xyz={CAMERA_XYZ} rpy={CAMERA_RPY}')
    if real_center_xyz is not None:
        print('Magenta real camera center relative to link6: '
              + ' '.join(f'{value:+.6f}' for value in real_center_xyz) + ' m')


if __name__ == '__main__':
    main()
