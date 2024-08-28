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
import os

'''
after run reset.py 
"odom frame"
child_frame_id: "camera_link"
pose: 
  pose: 
    position: 
      x: -0.0004420131735969335
      y: -0.00038160476833581924
      z: -0.0013358555734157562
    orientation: 
      x: -0.0054396671612655535
      y: 0.7049875439653867
      z: 0.00431244004409534
      w: 0.7091857131497313
'''

if __name__ == "__main__":
    rospy.init_node('ik_step', anonymous=True)
    robot = init_robot("robot2")
    robot2 = MyIK() 
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
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('s'):
            image_saver.record()
            slam_pose = slam_pose_listener.get_pose()
            joints = robot.get_joints()
            robot_pose = robot2.fk(joints)
            # slam_pose = cam_pose
            image_saver.record()  # Save images
            slam_poses.append(slam_pose)
            robot_poses.append(robot_pose)
            traj_poses.append(joints)
            
            print('robot pose', np.round(robot_pose[:3], 3))
            print("slam pose", np.round(slam_pose[:3], 3))            
        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            robot.step(action=lookup_action(code), wait=False)
        # save data
    ik = MyIK()
    robot_poses = np.array(robot_poses)
    slam_poses=np.array(slam_poses)
    traj_poses = np.array(traj_poses)

    np.save(os.path.join(home,  'robot_poses'), robot_poses)
    np.save(os.path.join(home,  'slam_poses'), slam_poses)
    np.save(os.path.join(home,  'traj'), traj_poses)


