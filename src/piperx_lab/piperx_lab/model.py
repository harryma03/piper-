"""Read Piper joint limits from the official upstream URDF."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET


ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))


def read_joint_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    root = ET.parse(urdf_path).getroot()
    if root.get('name') != 'piper':
        raise ValueError(f'Expected Piper URDF, got {root.get("name")!r}')
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    limits = {}
    for name in ARM_JOINTS:
        joint = joints.get(name)
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError(f'Missing revolute joint {name}')
        limit = joint.find('limit')
        if limit is None:
            raise ValueError(f'Missing limit for {name}')
        low, high = float(limit.get('lower')), float(limit.get('upper'))
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f'Invalid limit for {name}: {low}, {high}')
        limits[name] = (low, high)
    return limits


def installed_urdf_path() -> Path:
    from ament_index_python.packages import get_package_share_directory

    return (Path(get_package_share_directory('agx_arm_description')) /
            'agx_arm_urdf/piper/urdf/piper_description.urdf')


def installed_gripper_model_path() -> Path:
    return installed_urdf_path().with_name('piper_with_gripper_description.xacro')


def read_gripper_limit(model_path: Path) -> tuple[float, float]:
    root = ET.parse(model_path).getroot()
    joint = next((j for j in root.findall('joint') if j.get('name') == 'gripper'), None)
    if joint is None or joint.get('type') != 'prismatic':
        raise ValueError('Missing AgileX gripper joint')
    limit = joint.find('limit')
    if limit is None:
        raise ValueError('Missing AgileX gripper limit')
    low, high = float(limit.get('lower')), float(limit.get('upper'))
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high):
        raise ValueError('Invalid AgileX gripper limit')
    return low, high
