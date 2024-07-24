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


if __name__ == "__main__":
    rospy.init_node('ik_step', anonymous=True)
    robot = init_real_robot()   
    image_saver = MyImageSaver()
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    framedelay = 1000//20
    slam_pose_listener = RTABMapPoseListener(verbose=False)
    home = image_saver.folder_path

    # collect data
    slam_poses = []
    robot_poses = []

    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        cam_pose = get_cam_pose(frame, intrinsics)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('s'):
            image_saver.record()

            #slam_pose = slam_pose_listener.get_pose()
            slam_pose = cam_pose
            image_saver.record()  # Save images
            slam_poses.append(slam_pose)
            robot_poses.append(robot_pose)
            print('robot pose', np.round(robot_pose[:3], 3))
            print("cam pose", np.round(cam_pose[:3], 3))            
        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            robot_pose = step(robot, action=lookup_action(code), wait=True)
            
            

        # save data
    ik = MyIK()
    robot_poses = np.array(robot_poses)
    slam_poses=np.array(slam_poses)

    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'slam_poses'), slam_poses)

