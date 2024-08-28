#!/usr/bin/env python3
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from myRobotNs import MyRobotNs
import numpy as np
import math
import click

def deg2rad(deg):
    return [math.radians(degree) for degree in deg]

def rad2deg(radians):
    # Convert radians to degrees for each joint value
    return [math.degrees(rad) for rad in radians]

def swap_order(i, j, k):
    i[j], i[k] = i[k], i[j]
    return i

def reverse_sign(i, j):
    i[j] = -i[j]
    return i

def shake(robot, shake_delta):
    joint_degrees = rad2deg(robot.get_joints())
    print(joint_degrees)
    for i in range(len(joint_degrees)):
        for j in [-1, 1]:
            joint_degrees[i] += j * shake_delta
            joint_positions = deg2rad(joint_degrees)
            print('desired rads', joint_positions)
            print('current rads', robot.get_joints())
            robot.move_joints(joint_positions , duration=0.1, wait=False)
            rospy.sleep(0.5)  # Sleep for 0.5 seconds between movements

@click.command()
@click.argument('robot_name', type=click.Choice(['robot1', 'robot2'], case_sensitive=False))
def main(robot_name):
    try:
        rospy.init_node('shake', anonymous=True)
        robot = MyRobotNs(ns=robot_name)  # Initialize the robot object
        shake(robot, shake_delta=2)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()