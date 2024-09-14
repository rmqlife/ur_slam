#!/usr/bin/env python3

import rospy
import click
from myRobotNs import MyRobotNs
from myConfig import MyConfig  # Import the MyConfig class

@click.command()
@click.argument('robot_name', type=click.Choice(['robot1', 'robot2'], case_sensitive=False))
@click.option('--config', default='facedown', help='Name of the configuration to use (default is "facedown").')
def main(robot_name, config):
    """
    Initializes the robot and loads the specified joint configuration to reset the robot's pose.
    """
    try:
        rospy.init_node('reset', anonymous=True)
        my_robot = MyRobotNs(ns=robot_name)  # Initialize the robot object

        # Create a JointConfig instance
        joint_configs = MyConfig('../slam_data/joint_configs.json')

        # Check if the reset pose is already saved, and load it if available
        if joint_configs.get(config):
            reset_pose = joint_configs.get(config)
            print("Loaded reset pose:", reset_pose)
            my_robot.move_joints_smooth(reset_pose, coef=3, wait=False)
        else:
            print(f"Configuration '{config}' not found in joint configurations.")
    
    except rospy.ROSInterruptException:
        print('ROS Interrupt Exception occurred. Exiting...')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
