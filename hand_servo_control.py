import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from gazebo_msgs.srv import DeleteEntity, SpawnEntity

import cv2
import mediapipe as mp
import numpy as np
import os


class HandJointController(Node):

    def __init__(self):
        super().__init__('hand_joint_controller')

        # -------- Joint Publisher --------
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # -------- Gripper --------
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        # -------- Gazebo Services --------
        self.delete_client = self.create_client(
            DeleteEntity,
            '/gazebo/delete_entity'
        )

        self.spawn_client = self.create_client(
            SpawnEntity,
            '/gazebo/spawn_entity'
        )

        # -------- Path to box.sdf --------
        self.box_path = os.path.expanduser(
            '~/ros2_ws/src/open_manipulator/open_manipulator_x_bringup/worlds/box.sdf'
        )

        # -------- MediaPipe --------
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)

        # -------- Joint State --------
        self.joint_positions = [0.0, -0.5, 0.5]
        self.prev_grip_state = None

        self.timer = self.create_timer(0.03, self.process_frame)

    # ------------------------------------------------

    def send_trajectory(self):

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()
        point.positions = [
            self.joint_positions[0],
            self.joint_positions[1],
            self.joint_positions[2],
            0.0   # joint4 คงที่
        ]

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.6 * 1e9)

        traj.points.append(point)
        self.traj_pub.publish(traj)

    # ------------------------------------------------

    def send_gripper_goal(self, position):

        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            return

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 80.0  # เพิ่มแรงบีบ

        self.gripper_client.send_goal_async(goal)

    # ------------------------------------------------

    def reset_box(self):

        if not self.delete_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Delete service not available")
            return

        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Spawn service not available")
            return

        # ----- Delete -----
        del_req = DeleteEntity.Request()
        del_req.name = "my_box"
        self.delete_client.call_async(del_req)

        # ----- Spawn -----
        spawn_req = SpawnEntity.Request()
        spawn_req.name = "my_box"

        try:
            with open(self.box_path, 'r') as f:
                spawn_req.xml = f.read()
        except:
            self.get_logger().error("box.sdf not found")
            return

        spawn_req.initial_pose.position.x = 0.7
        spawn_req.initial_pose.position.y = 0.0
        spawn_req.initial_pose.position.z = 0.3
        spawn_req.initial_pose.orientation.w = 1.0

        self.spawn_client.call_async(spawn_req)

        self.get_logger().info("Box Reset Complete")

    # ------------------------------------------------

    def process_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

            palm = hand.landmark[9]

            x = (palm.x - 0.5) * 3.5
            y = (palm.y - 0.5) * 3.5

            self.joint_positions[0] = np.clip(x * 2.0, -2.0, 2.0)
            self.joint_positions[1] = np.clip(y * 2.0, -2.0, 2.0)
            self.joint_positions[2] = np.clip(-y * 2.0, -2.0, 2.0)

            self.send_trajectory()

            # ----- Gripper Control -----
            tips = [8, 12, 16, 20]
            finger_count = 0

            for tip in tips:
                if hand.landmark[tip].y < hand.landmark[tip - 2].y:
                    finger_count += 1

            if finger_count >= 3:
                grip_state = "open"
            else:
                grip_state = "close"

            if grip_state != self.prev_grip_state:

                if grip_state == "open":
                    self.send_gripper_goal(0.018)
                else:
                    self.send_gripper_goal(0.004)

                self.prev_grip_state = grip_state

        cv2.imshow("Hand Control (Press R to reset box)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            self.reset_box()

    # ------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = HandJointController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()