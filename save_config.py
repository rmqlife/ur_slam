import click
from ros_utils.myConfig import MyConfig
from ros_utils.myRobotNs import MyRobotNs
import rospy
@click.command()
@click.argument('robot_name', type=click.Choice(['robot1', 'robot2'], case_sensitive=False))
@click.argument('config_name', required=True, default=None)


def main(robot_name, config_name):
    try:
        # Initialize the ROS node
        rospy.init_node('ur5_save_config', anonymous=True)
        
        # Initialize the robot with the specified robot_name
        robot = MyRobotNs(ns=robot_name) 
        joint_configs = MyConfig('slam_data/joint_configs.json')
        
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
