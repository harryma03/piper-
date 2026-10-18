from setuptools import setup

setup(
    name='piperx_lab',
    version='0.1.0',
    packages=['piperx_lab'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/piperx_lab']),
        ('share/piperx_lab', ['package.xml']),
        ('share/piperx_lab/calibration', ['calibration/board.json',
                                         'calibration/board.png']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='co',
    maintainer_email='co@example.com',
    description='Piper project control tools',
    license='MIT',
    entry_points={
        'console_scripts': [
            'joint_move = piperx_lab.joint_move:main',
            'camera_target = piperx_lab.camera_target:main',
            'gripper_move = piperx_lab.gripper_move:main',
            'handeye_capture = piperx_lab.handeye_capture:main',
            'handeye_solve = piperx_lab.handeye_solve:main',
            'target_base = piperx_lab.target_base:main',
            'target_check = piperx_lab.target_check:main',
            'cube_pose = piperx_lab.cube_pose:main',
            'board_check = piperx_lab.board_check:main',
        ],
    },
)
