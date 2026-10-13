"""Generate a Piper Gazebo model with a nominal link6-mounted camera.

The arm and gripper come from the official agx_arm_description xacro. The
camera transform is simulation ground truth, not a measurement of the real
mount.
"""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro


CAMERA_XYZ = '0.035 0 -0.045'
CAMERA_RPY = '3.14157 0.19916 0'


def add(parent, tag, attributes=None, text=None):
    child = ET.SubElement(parent, tag, attributes or {})
    if text is not None:
        child.text = str(text)
    return child


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    share = Path(get_package_share_directory('agx_arm_description'))
    source = share / 'agx_arm_urdf/piper/urdf/piper_with_gripper_description.xacro'
    root = ET.fromstring(xacro.process_file(str(source)).toxml())
    prefix = 'package://agx_arm_description/'
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename', '')
        if filename.startswith(prefix):
            mesh.set('filename', (share / filename[len(prefix):]).resolve().as_uri())

    camera = add(root, 'link', {'name': 'camera_link'})
    inertial = add(camera, 'inertial')
    add(inertial, 'mass', {'value': '0.08'})
    add(inertial, 'inertia', {'ixx': '0.00002', 'ixy': '0', 'ixz': '0',
                              'iyy': '0.00002', 'iyz': '0', 'izz': '0.00002'})
    visual = add(camera, 'visual')
    geometry = add(visual, 'geometry')
    add(geometry, 'box', {'size': '0.045 0.06 0.035'})
    material = add(visual, 'material', {'name': 'sim_camera_blue'})
    add(material, 'color', {'rgba': '0.1 0.35 0.9 1'})
    collision = add(camera, 'collision')
    add(add(collision, 'geometry'), 'box', {'size': '0.045 0.06 0.035'})

    joint = add(root, 'joint', {'name': 'sim_camera_mount', 'type': 'fixed'})
    add(joint, 'parent', {'link': 'link6'})
    add(joint, 'child', {'link': 'camera_link'})
    add(joint, 'origin', {'xyz': CAMERA_XYZ, 'rpy': CAMERA_RPY})
    add(root, 'link', {'name': 'camera_depth_frame'})
    optical = add(root, 'joint', {'name': 'sim_camera_optical', 'type': 'fixed'})
    add(optical, 'parent', {'link': 'camera_link'})
    add(optical, 'child', {'link': 'camera_depth_frame'})
    add(optical, 'origin', {'xyz': '0 0 0', 'rpy': '-1.5707963 0 -1.5707963'})

    for link in root.findall('link'):
        gazebo = add(root, 'gazebo', {'reference': link.attrib['name']})
        add(gazebo, 'gravity', text='false')
    preserve = add(root, 'gazebo', {'reference': 'sim_camera_mount'})
    add(preserve, 'preserveFixedJoint', text='true')

    gazebo = add(root, 'gazebo', {'reference': 'camera_link'})
    sensor = add(gazebo, 'sensor', {'name': 'handeye_color', 'type': 'camera'})
    add(sensor, 'always_on', text='true')
    add(sensor, 'update_rate', text='15')
    camera_sensor = add(sensor, 'camera', {'name': 'handeye_color'})
    add(camera_sensor, 'horizontal_fov', text='1.858776')
    image = add(camera_sensor, 'image')
    add(image, 'width', text='640')
    add(image, 'height', text='480')
    add(image, 'format', text='R8G8B8')
    clip = add(camera_sensor, 'clip')
    add(clip, 'near', text='0.02')
    add(clip, 'far', text='3.0')
    plugin = add(sensor, 'plugin', {'name': 'handeye_camera_ros',
                                    'filename': 'libgazebo_ros_camera.so'})
    ros = add(plugin, 'ros')
    add(ros, 'namespace', text='/sim/camera')
    add(plugin, 'camera_name', text='aligned')
    add(plugin, 'frame_name', text='camera_depth_frame')

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
    print(f'Saved {args.output}; nominal T_link6_camera_link xyz={CAMERA_XYZ} '
          f'rpy={CAMERA_RPY}')


if __name__ == '__main__':
    main()
