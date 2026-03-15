#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

from linkattacher_msgs.srv import AttachLink, DetachLink

import cv2
import mediapipe as mp
import numpy as np


class HandJointController(Node):

    def __init__(self):
        super().__init__('hand_joint_controller')

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_controller/gripper_cmd'
        )

        self.attach_client = self.create_client(AttachLink, '/ATTACHLINK')
        self.detach_client = self.create_client(DetachLink, '/DETACHLINK')

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)

        self.joint_positions = [0.0, -0.5, 0.5]
        self.joint4 = 0.0

        self.prev_grip_state = None
        self.thumb_trigger = False

        self.timer = self.create_timer(0.03, self.process_frame)

    # -------------------------------------------------

    def send_trajectory(self):

        traj = JointTrajectory()
        traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4']

        point = JointTrajectoryPoint()

        point.positions = [
            self.joint_positions[0],
            self.joint_positions[1],
            self.joint_positions[2],
            self.joint4
        ]

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = int(0.4 * 1e9)

        traj.points.append(point)

        self.traj_pub.publish(traj)

    # -------------------------------------------------

    def send_gripper_goal(self, position):

        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            return

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 7.0

        self.gripper_client.send_goal_async(goal)

    # -------------------------------------------------

    def attach_box(self):

        if not self.attach_client.wait_for_service(timeout_sec=1.0):
            return

        req = AttachLink.Request()

        req.model1_name = 'open_manipulator_x_system'
        req.link1_name = 'gripper_right_link'

        req.model2_name = 'my_box'
        req.link2_name = 'box_link'

        self.attach_client.call_async(req)

    # -------------------------------------------------

    def detach_box(self):

        if not self.detach_client.wait_for_service(timeout_sec=1.0):
            return

        req = DetachLink.Request()

        req.model1_name = 'open_manipulator_x_system'
        req.link1_name = 'gripper_right_link'

        req.model2_name = 'my_box'
        req.link2_name = 'box_link'

        self.detach_client.call_async(req)

    # -------------------------------------------------

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

            x = (palm.x - 0.5) * 2.0
            y = (palm.y - 0.5) * 2.0

            self.joint_positions[0] = np.clip(x * 2.0, -2.0, 2.0)
            self.joint_positions[1] = np.clip(y * 2.0, -2.0, 2.0)
            self.joint_positions[2] = np.clip(-y * 2.0, -2.0, 2.0)

            # -------------------------
            # Thumb control joint4
            # -------------------------

            thumb_tip = hand.landmark[4]
            thumb_ip = hand.landmark[3]

            thumb_up = thumb_tip.y < thumb_ip.y

            if thumb_up and not self.thumb_trigger:

                self.joint4 += 0.2
                self.joint4 = np.clip(self.joint4, -1.7, 1.7)

                self.thumb_trigger = True

            if not thumb_up:
                self.thumb_trigger = False

            self.send_trajectory()

            # -------------------------
            # Finger counting
            # -------------------------

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
                    self.detach_box()

                else:

                    self.send_gripper_goal(0.00145)
                    self.attach_box()

                self.prev_grip_state = grip_state

        cv2.imshow("Hand Control", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            rclpy.shutdown()

        if key == ord('r'):

            self.joint4 = 0.0
            self.send_trajectory()

            print("Joint4 Reset")

    # -------------------------------------------------


def main(args=None):

    rclpy.init(args=args)

    node = HandJointController()

    rclpy.spin(node)

    node.cap.release()
    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
