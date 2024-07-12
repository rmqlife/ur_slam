from utils.aruco_util import *
from ros_utils.myImageSaver import MyImageSaver
import rospy
import cv2
from replay_aruco_poses import *

def get_aruco_poses(corners, ids, intrinsics):
    # make sure the aruco's orientation in the camera view! 
    poses = estimate_markers_poses(corners, marker_size=0.03, intrinsics=intrinsics)  # Marker size in meters

    poses_dict = {}
    # detected
    if ids is not None:
        for k, iden in enumerate(ids):
            poses_dict[iden]=poses[k] 

    return poses_dict


if __name__=="__main__":
    rospy.init_node('image_saver')

    image_saver = MyImageSaver()
    framedelay = 1000//20

    instrinsics = load_intrinsics("slam_data/intrinsics_d435.json")

    init_pose = None
    #image_saver.record()  # Save images
    while not rospy.is_shutdown():
        
        frame = image_saver.rgb_image
        corners, ids = detect_aruco(frame, draw_flag=True)# 

        cv2.imshow('Camera', frame)
        # Exit on 'q' key press
        if cv2.waitKey(framedelay) & 0xFF == ord('q'):
            break
        
        poses_dict = get_aruco_poses(corners=corners, ids=ids, intrinsics=instrinsics)

        id = 0
        if id in poses_dict:
            current_pose = poses_dict[id]
            if init_pose is None:
                init_pose = current_pose
            else:
                # compute the R, t
                init_cam, current_cam = reverse_poses([init_pose, current_pose])
                # compute 
                cam_delta = pose_delta(current_cam, init_cam)
                print('cam', np.round(cam_delta[:3], 3))

    cv2.destroyAllWindows()