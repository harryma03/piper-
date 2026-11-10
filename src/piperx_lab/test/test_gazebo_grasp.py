"""Simulation backend identity and generated control-model regression checks."""
import os
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]


def test_grasp_runner_refuses_real_domain():
    result = subprocess.run([sys.executable, str(ROOT/'sim/gazebo/grasp/run_grasp.py')],
                            env={**os.environ, 'ROS_DOMAIN_ID': '0'}, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Requires isolated ROS_DOMAIN_ID=183' in result.stderr


def test_generated_contact_model_separates_arm_and_hand(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIPER_SIM_')}
    subprocess.run([sys.executable, str(ROOT/'sim/gazebo/grasp/make_scene.py'),str(tmp_path)],check=True,capture_output=True,env=env)
    robot=ET.parse(tmp_path/'robot.urdf').getroot()
    arm=robot.find("ros2_control[@name='GazeboArm']")
    hand=robot.find("ros2_control[@name='GazeboHand']")
    assert {j.get('name') for j in arm.findall('joint')} == {f'joint{i}' for i in range(1,7)}
    assert {j.get('name') for j in hand.findall('joint')} == {'gripper','gripper_joint1','gripper_joint2'}
    for name in ['gripper_joint1','gripper_joint2']:
        assert robot.find(f"joint[@name='{name}']/mimic") is None
        assert hand.find(f"joint[@name='{name}']/param[@name='pos_kp']") is not None
    world=ET.parse(tmp_path/'grasp.world').getroot()
    cylinder=world.find("world/model[@name='cylinder']")
    assert cylinder.find('static') is None or cylinder.find('static').text == 'false'
    assert float(cylinder.find('.//collision/geometry/cylinder/radius').text) == 0.023
    assert not any('attach' in p.get('filename','').lower() for p in world.findall('.//plugin'))
    parameters = json.loads((tmp_path/'scene_parameters.json').read_text())
    assert parameters['finger_geometry'] == 'mesh'
    assert parameters['friction'] == 2
    assert parameters['finger_kp'] == 2000
    assert float(cylinder.find('.//friction/ode/mu').text) == parameters['friction']


def test_simulation_experiment_parameters_are_recorded(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIPER_SIM_')}
    env.update(PIPER_SIM_FRICTION='3', PIPER_SIM_FINGER_GEOMETRY='box')
    subprocess.run([sys.executable, str(ROOT/'sim/gazebo/grasp/make_scene.py'),str(tmp_path)],check=True,capture_output=True,env=env)
    parameters = json.loads((tmp_path/'scene_parameters.json').read_text())
    assert parameters['friction'] == 3
    assert parameters['finger_geometry'] == 'box'
    robot = ET.parse(tmp_path/'robot.urdf').getroot()
    assert robot.find("link[@name='gripper_link1']/collision/geometry/box") is not None


def test_grasp_success_requires_contacts_during_close_and_lift():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gazebo_grasp_runner", ROOT / "sim/gazebo/grasp/run_grasp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    close = {"gripper_link1": True, "gripper_link2": True}
    assert not module.both_fingers_in_both_phases({"closing": close})
    assert not module.both_fingers_in_both_phases({"closing": close, "lifting": {"gripper_link1": True}})
    assert module.both_fingers_in_both_phases({"closing": close, "lifting": close})


def test_rgbd_scene_keeps_camera_and_randomizes_target_without_metadata_truth(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIPER_SIM_')}
    env.update(PIPER_SIM_RGBD='1', PIPER_SIM_OBJECT_X='.42', PIPER_SIM_OBJECT_Y='-.03')
    subprocess.run([sys.executable, str(ROOT/'sim/gazebo/grasp/make_scene.py'), str(tmp_path)],
                   check=True, capture_output=True, env=env)
    robot = ET.parse(tmp_path/'robot.urdf').getroot()
    assert robot.find("gazebo[@reference='camera_link']/sensor[@type='depth']") is not None
    world = ET.parse(tmp_path/'grasp.world').getroot()
    pose = world.find("world/model[@name='cylinder']/pose").text.split()
    assert float(pose[0]) == .42 and float(pose[1]) == -.03
    assert 'object_initial_xy_m' not in json.loads((tmp_path/'scene_parameters.json').read_text())


def test_vision_fit_recovers_unknown_position_with_explicit_priors():
    import importlib.util
    import numpy as np
    from scipy.spatial.transform import Rotation
    spec = importlib.util.spec_from_file_location('sim_observe', ROOT/'sim/gazebo/grasp/observe_cylinder.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(8)
    angles, heights = np.meshgrid(np.linspace(-1.1, 1.1, 30), np.linspace(-.13, -.08, 20))
    points_base = np.column_stack((.43 + .023*np.cos(angles).ravel(),
                                   -.028 + .023*np.sin(angles).ravel(), heights.ravel()))
    transform = np.eye(4)
    transform[:3,:3] = Rotation.from_euler('xyz', [.2,-.3,.1]).as_matrix()
    transform[:3,3] = [.2,.05,.3]
    cloud = (points_base-transform[:3,3]) @ transform[:3,:3]
    cloud += rng.normal(0, .0001, cloud.shape)
    result = module.observation_from_cloud(cloud, transform)
    np.testing.assert_allclose(result['center_xyz_m'], [.43, -.028, -.11], atol=.001)
    assert result['source'] == 'sim_rgbd_red_cylinder_fit'
    assert result['priors']['upright']


def test_visual_planner_rejects_failed_or_nonfinite_observation():
    import importlib.util
    import pytest
    spec = importlib.util.spec_from_file_location('sim_grasp', ROOT/'sim/gazebo/grasp/run_grasp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = {'status':'ok','source':'sim_rgbd_red_cylinder_fit','base_frame':'base_link','center_xyz_m':[.43,-.02,-.11]}
    assert module.planning_center_from_observation(valid) == [.43,-.02,-.11]
    with pytest.raises(ValueError):
        module.planning_center_from_observation({**valid, 'status':'failed'})
    with pytest.raises(ValueError):
        module.planning_center_from_observation({**valid, 'center_xyz_m':[float('nan'),0.,0.]})
