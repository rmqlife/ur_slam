#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from util import rad2deg, deg2rad, swap_order, reverse_sign

from myOmni import MyOmni
from ik_step import *


if __name__ == '__main__':
    try:
        rospy.init_node('apply_delta', anonymous=True)
        my_omni = MyOmni()  # Initialize the omni object
        my_robot = init_robot()  # Initialize the robot object
        teleop_sign = False
        hold_to_control=False
    
        print('press gray button to start function')
        while not rospy.is_shutdown():

            teleop_sign_prev = teleop_sign
            if hold_to_control:
                teleop_sign = my_omni.gray_button
            else:
                teleop_sign = my_omni.gray_button_flag
            
            # Check if the gray button state has changed and recording is not active

            omni_pose = pose_to_SE3(my_omni.pose)
            # swap x and y axis, make the y = -y
            transform = SE3([[0,-1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
            omni_pose = transform * omni_pose

            robot_pose = pose_to_SE3(my_robot.get_pose())

            if teleop_sign and not teleop_sign_prev:
                my_robot.start_bag_recording()
                initial_omni_pose = omni_pose.copy()
                initial_robot_pose = robot_pose.copy()

            # Check if the gray button state has changed and recording is active
            elif not teleop_sign and teleop_sign_prev:
                my_robot.stop_bag_recording()
                
            # Print joint states while recording is active
            if teleop_sign:
                # print("robot at", rad2deg(my_robot.get_pose()))
                # print("omni at", rad2deg(my_omni.pose))
                delta_omni_pose =  omni_pose * initial_omni_pose.inv()
                print('omni delta is')
                delta_omni_pose.printline()
                
                initial_robot_pose.printline()
                goal_robot_pose = delta_omni_pose * initial_robot_pose
                goal_robot_pose.t =  3*(delta_omni_pose.t) + initial_robot_pose.t


                my_robot.goto_pose(goal_robot_pose, wait=False, coef=1, joint_thresh=3.14)

                # robot_joints = initial_robot_joints + delta_omni_joints
                # # compute a distance 
                # dist = np.linalg.norm(delta_omni_joints)
                # print('distance', dist)
                # my_robot.move_joints(robot_joints, duration=0.1)
            
            # control the gripper
            if my_omni.white_button_flag:
                my_omni.white_button_flag = not my_omni.white_button_flag
                print('clicked white button')

            rospy.sleep(0.01)

    except rospy.ROSInterruptException:
        pass
