from setuptools import find_packages, setup

package_name = 'hand_servo_control'

setup(
    name='hand_servo_control',
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/hand_servo_control', ['hand_landmarker.task']),
        ('share/' + package_name + '/launch', ['launch/full_system.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='weerawat',
    maintainer_email='weerawat@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hand_tracking = hand_servo_control.hand_tracking_node:main',
            'hand_mapper = hand_servo_control.hand_to_servo_mapper:main',
            'openmanipulator_controller=hand_servo_control.openmanipulator_controller:main',
            'hand_servo_controller = hand_servo_control.hand_servo_controller:main',
            'hand_tracking_node4 = hand_servo_control.hand_tracking_node4:main',
            'hand_tracking_node2 = hand_servo_control.hand_tracking_node2:main',
            'hand_servo3 = hand_servo_control.hand_servo3:main',
        ],
    },
)
