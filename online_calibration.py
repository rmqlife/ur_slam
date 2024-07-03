from hand_eye_slam import HandEyeSlam
import os
import numpy as np
from myIK_SLAM import init_real_robot
from myIK import MyIK
from pose_util import *
from myImageSaver import MyImageSaver
from myRTABMap import RTABMapPoseListener


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
    poses = myIK.forward_joints(joints)
    print(poses)
    # plan poses
    traj = myIK.plan_trajectory(poses, joints[0], vel_threshold=vel_threshold)
    return traj

def hand_eye_calibration():
    # upright    
    traj = configs_to_traj("slam_data/joint_configs.json", vel_threshold=0.03)
    traj = traj.tolist()

    # init
    robot = init_real_robot()
    slam_pose_listener = RTABMapPoseListener(verbose=False)
    image_saver = MyImageSaver()
    home = image_saver.folder_path
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
    robot_poses = ik.forward_joints(traj)
    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'traj'), traj)
    np.save(os.path.join(home,  'slam_poses'), slam_poses)

    # run hand eye slam
    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.estimate(slam_poses, robot_poses)
    new_poses = hand_eye_slam.slam_to_robot(slam_poses, verbose=1)
    print("overall projected error", poses_error(robot_poses, new_poses))
    hand_eye_slam.save(os.path.join(home, 'hand_eye_slam.npz'))
    return 

if __name__=="__main__":
    hand_eye_calibration()