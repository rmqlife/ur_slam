import numpy as np
import roboticstoolbox as rtb
from utils.pose_util import *
from spatialmath import SE3, SO3
from myIK import MyIK
import rospy
from std_msgs.msg import Float32
import cv2
from follow_aruco import *
from ros_utils.myImageSaver import MyImageSaver
from ros_utils.myRTABMap import RTABMapPoseListener
from ik_step import *

def configs_to_traj(config_path, vel_threshold):
    from myConfig import MyConfig
    joint_configs = MyConfig(config_path)
    # print(joint_configs.config_dict)
    joints = []
    for k in ['q1','q2','q3','q4','q1']:
        j = joint_configs.get(k)
        print(j)
        joints.append(j)

    myIK = MyIK()
    poses = myIK.fk(joints)

    # plan poses
    traj = myIK.plan_trajectory(poses, joints[0], vel_threshold=vel_threshold)
    return traj

def fixed_path_hand_eye_calibration():
    # upright    
    traj = configs_to_traj("slam_data/joint_configs.json", vel_threshold=0.03)
    traj = traj.tolist()

    # init
    robot = init_robot()
    slam_pose_listener = RTABMapPoseListener(verbose=False)
    image_saver = MyImageSaver()
    home = image_saver.folder_path
    print("saving at", home)
    robot.move_joints(traj[0], 2.0, wait=True)

    # collect data
    slam_poses = []
    for joints in traj[:]:
        print("moving to", joints)
        # keep joints-end
        robot.move_joints(joints, 0.3, wait=True)
        image_saver.record()  # Save images
        slam_poses.append(slam_pose_listener.get_pose())
    slam_poses=np.array(slam_poses)

    # save data
    ik = MyIK()
    robot_poses = ik.fk(traj)
    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'traj'), traj)
    np.save(os.path.join(home,  'slam_poses'), slam_poses)

    return 

if __name__=="__main__":
    fixed_path_hand_eye_calibration()