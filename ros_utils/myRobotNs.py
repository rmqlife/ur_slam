#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import numpy as np

class MyRobotNs:  # Inherit from MyBag
    def __init__(self,ns="robot1"):
        '''
        ns : name space , String
        '''
        self.topic = '/'+ns+'/scaled_pos_joint_traj_controller/state'  # Define the topic for reading joint states

        # Create a subscriber to the '/joint_states' topic
        self.robot_joint_subscriber = rospy.Subscriber(self.topic, JointTrajectoryControllerState, self.subscriber_callback)
        pub_topic = '/'+ns+'/scaled_pos_joint_traj_controller/command'
        self.pub = rospy.Publisher(pub_topic, JointTrajectory, queue_size=10)

        # Initialize joint names and positions
        self.current_joint_positions = []

        # Wait for the subscriber to receive joint names
        rospy.sleep(0.5)

    def subscriber_callback(self, data):
        self.joint_positions = np.array(data.actual.positions)
        self.joint_names = data.joint_names

    def get_joints(self):
        return self.joint_positions
    

    def move_joints_smooth(self, joints, coef=3, wait=False):
        joints_movement = np.max(np.abs(joints - self.get_joints()))
        return self.move_joints(joints, duration=coef*joints_movement, wait=wait)


    def move_joints(self, joint_positions, duration=0.1, wait=True):
        # Create a JointTrajectory message
        joint_traj = JointTrajectory()

        # Set the joint names from the received message
        joint_traj.joint_names = self.joint_names

        # Create a JointTrajectoryPoint for the desired joint positions
        point = JointTrajectoryPoint()
        point.positions = joint_positions

        # Set the time from start for this point
        point.time_from_start = rospy.Duration(duration)

        # Append the JointTrajectoryPoint to the trajectory
        joint_traj.points.append(point)
        self.pub.publish(joint_traj)  # Call the publish method

        threshold = 1e-4
        if wait:
            # Check if joint positions have been reached
            while not rospy.is_shutdown():
                current_positions = self.get_joints()
                if np.all(np.abs(current_positions - joint_positions) < threshold):
                    break
                # print('wait', np.abs(current_positions - joint_positions))
                rospy.sleep(0.001)  # Adjust sleep duration as needed
        else:
            rospy.sleep(duration)



