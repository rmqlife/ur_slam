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

# USAGE: 
# traj = configs_to_traj("slam_data/joint_configs.json", vel_threshold=0.03)
# traj = traj.tolist()
def configs_to_traj(config_path, vel_threshold):
    from myConfig import MyConfig
    joint_configs = MyConfig(config_path)
    # print(joint_configs.config_dict)
    joints = []
    poses = []
    myIK = MyIK()
    for k in ['q1','q2','q3','q4','q1']:#for robot1
    # for k in ['k1','k2','k3','k4','k5','k6','k1']:#for robot2
        j = joint_configs.get(k)
        print(j)
        poses.append(myIK.fk(j))
        joints.append(j)
    # plan poses
    traj = myIK.plan_trajectory(poses, joints[0], vel_threshold=vel_threshold)
    return traj


if __name__ == "__main__":
    traj = configs_to_traj("slam_data/joint_configs.json", vel_threshold=0.03)
    traj = traj.tolist()
    rospy.init_node('ik_step', anonymous=True)
    chosen_robot = 'robot1'
    robot = init_robot(chosen_robot)
    robot_without_basetrans = MyIK() 
    image_saver = MyImageSaver()
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    framedelay = 1000//20
    slam_pose_listener = RTABMapPoseListener(verbose=False)
    home = image_saver.folder_path

    # collect data
    slam_poses = []
    robot_poses = []
    traj_poses = []

    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        cam_pose = get_cam_pose(frame, intrinsics)

        for traj_path in traj:
            robot.move_joints_smooth(traj_path, coef=3, wait=True)
            # input("press enter to save")
            image_saver.record()
            slam_pose = slam_pose_listener.get_pose()
            joints = robot.get_joints()
            robot_pose = robot_without_basetrans.fk(joints)
            # slam_pose = cam_pose
            image_saver.record()  # Save images
            slam_poses.append(slam_pose)
            robot_poses.append(robot_pose)
            traj_poses.append(joints)
            
            print('robot pose', np.round(robot_pose[:3], 3))
            print("slam pose", np.round(slam_pose[:3], 3))
        break            
    # save data

    ik = MyIK()
    robot_poses = np.array(robot_poses)
    slam_poses=np.array(slam_poses)
    traj_poses = np.array(traj_poses)

    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'slam_poses'), slam_poses)
    np.save(os.path.join(home,  'traj'), traj_poses)



