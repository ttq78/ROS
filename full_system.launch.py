from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # --------- Paths ---------
    bringup_pkg = get_package_share_directory('open_manipulator_x_bringup')
    moveit_pkg = get_package_share_directory('open_manipulator_x_moveit_config')

    gazebo_launch = os.path.join(bringup_pkg, 'launch', 'gazebo.launch.py')
    servo_launch = os.path.join(moveit_pkg, 'launch', 'servo.launch.py')

    # --------- Gazebo ---------
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch)
    )

    # --------- MoveIt Servo (Simulation Mode) ---------
    servo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(servo_launch),
        launch_arguments={'use_sim': 'true'}.items()
    )

    # --------- Start Servo (delay 3 sec) ---------
    start_servo = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/servo_node/start_servo',
                    'std_srvs/srv/Trigger',
                    '{}'
                ],
                output='screen'
            )
        ]
    )

    # --------- Hand Controller ---------
    hand_control = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='hand_servo_control',
                executable='hand_tracking_node2',
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        gazebo,
        servo,
        start_servo,
        hand_control
    ])
