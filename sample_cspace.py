import numpy as np
import roboticstoolbox as rtb
from utils.pose_util import *
from ik_step import init_robot
import rospy

# USAGE: 
# traj = configs_to_traj("slam_data/joint_configs.json", vel_threshold=0.03)
# traj = traj.tolist()
def load_configs(config_path):
    from myConfig import MyConfig
    joint_configs = MyConfig(config_path)
    # print(joint_configs.config_dict)
    joints = []
    for k in ['pmin','pmax']:
        j = joint_configs.get(k)
        print(j)
        joints.append(j)
    return joints

def bounding_box_points(pose_min, pose_max):
    # Extract positional and rotational components
    pos_min = np.array(pose_min.t)
    pos_max = np.array(pose_max.t)
    rot_min = np.array(pose_min.rpy( unit='deg'))
    rot_max = np.array(pose_max.rpy( unit='deg'))

    print(rot_min, rot_max)

    # Generate all combinations of positional and rotational components
    points = []
    for x in [pos_min[0], pos_max[0]]:
        for y in [pos_min[1], pos_max[1]]:
            for z in [pos_min[2], pos_max[2]]:
                for roll in [rot_min[0], rot_max[0]]:
                    for pitch in [rot_min[1], rot_max[1]]:
                        for yaw in [rot_min[2], rot_max[2]]:
                            # print([roll, pitch, yaw])
                            point = SE3.RPY(roll, pitch, yaw, unit='deg')
                            point.t = [x,y, z]
                            point.printline()
                            # print()
                            points.append(point)

    return points

if __name__ == "__main__":
    rospy.init_node('c_space', anonymous=True)
    robot = init_robot()
    
    joints = load_configs('slam_data/joint_configs.json')

    min_pose = robot.myIK.fk_se3(joints[0])
    max_pose = robot.myIK.fk_se3(joints[1])
    points = bounding_box_points(min_pose, max_pose)
    
    # print(points)
    for p in points:
        p.printline()
        robot.goto_pose(p, wait=True, coef=2, joint_thresh=3.14)
