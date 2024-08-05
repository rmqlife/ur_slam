from utils.aruco_util import *
from ros_utils.myImageSaver import MyImageSaver
import rospy
import cv2
# from replay_aruco_poses import *
from std_msgs.msg import Float32
import os
from ik_step import *

def get_aruco_poses(corners, ids, intrinsics):
    # make sure the aruco's orientation in the camera view! 
    poses = estimate_markers_poses(corners, marker_size=0.03, intrinsics=intrinsics)  # Marker size in meters
    poses_dict = {}
    # detected
    if ids is not None:
        for k, iden in enumerate(ids):
            poses_dict[iden]=poses[k] 

    return poses_dict

def get_cam_pose(frame, intrinsics):
    corners, ids = detect_aruco(frame, draw_flag=True)# 
    poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=intrinsics)
    id = 0
    current_cam = None
    if id in poses_dict:
        current_pose = poses_dict[id]
            # compute the R, t
        current_cam = inverse_pose(current_pose)
            # compute 
        # print('cam', np.round(current_cam[:3], 3))
    return current_cam




if __name__=="__main__":
    rospy.init_node('follow_aruco')
    image_saver = MyImageSaver()
    rospy.sleep(1)
    framedelay = 1000//20
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")

    robot = init_robot()

    goal_frame = None

    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        cam_pose = get_cam_pose(frame, intrinsics)
        if cam_pose is not None:
            cam_pose = pose_to_SE3(cam_pose)

        if goal_frame is None and cam_pose is not None:
            goal_frame = frame
            goal_pose = cam_pose

        cv2.imshow('Camera', frame)
        # Exit on 'q' key press
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('s'):
            image_saver.record()
        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            action = lookup_action(code)
            pose = robot.step(action=action, wait=False)
            print('robot pose', np.round(pose[:3], 3))
            if cam_pose is not None:
                print("cam pose")
                cam_pose.printline()
                move = goal_pose * cam_pose.inv()
                print('action todo')
                move.printline()
        if key == ord('m') and move is not None:
            robot.step(action=move, wait=False)
            
    cv2.destroyAllWindows()