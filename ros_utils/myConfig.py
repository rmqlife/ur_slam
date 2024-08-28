#!/usr/bin/env python3
import rospy
import json
from myRobotNs import MyRobotNs  # Import your MyRobot class
import click

class MyConfig:
    def __init__(self, file_path='joint_configs.json'):
        self.file_path = file_path
        self.load()  # Initialize the config_dict with loaded configurations
    
    def save(self, config_name, joint_positions):
        # Create a dictionary to store the joint positions with a name
        # Open the JSON file for writing
        self.config_dict[config_name]=list(joint_positions)
        print(self.config_dict)
        with open(self.file_path, 'w') as file:
            # Append the new joint configuration to the file
            json.dump(self.config_dict, file, indent=4)

    def load(self):
        try:
            # Open the JSON file for reading
            with open(self.file_path, 'r') as file:
                self.config_dict = json.load(file)
        except Exception as e:
            # Handle the case where the file does not exist
            print(f"Configuration file '{self.file_path}' not found.")
            self.config_dict ={}


    def get(self, name):
        return self.config_dict[name]
    
@click.command()
@click.argument('robot_name', type=click.Choice(['robot1', 'robot2'], case_sensitive=False))
@click.argument('config_name', required=True, default=None)
def main(robot_name, config_name):
    try:
        # Initialize the ROS node
        rospy.init_node('ur5_save_config', anonymous=True)
        
        # Initialize the robot with the specified robot_name
        robot = MyRobotNs(ns=robot_name) 
        joint_configs = MyConfig()
        
        print("Current configurations:", joint_configs.config_dict)
        
        joints = robot.get_joints()
        print("Current joint states:", joints)
        
        if config_name:
            # Specify the configuration name to save
            joint_configs.save(config_name, joints)
            print(f"Saved joint configuration '{config_name}': {joints}")
        else:
            print("No configuration name provided. Usage: ./your_script_name.py <robot_name> <config_name>")

    except rospy.ROSInterruptException:
        print('ROS Interrupt Exception occurred. Exiting...')
    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()
