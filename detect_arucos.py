import cv2
from utils.aruco_util import *

def capture_poses(src_path, dst_path, intrinsics_path="slam_data/intrinsics_d435.json"):
    cap = cv2.VideoCapture(src_path)  # Change to your camera index or video file path
    intrinsics = load_intrinsics(intrinsics_path)
    framedelay=1000//30
    # framedelay=1
    poses_traj = dict()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Estimate camera pose using ArUco markers
        corners, ids = detect_aruco(frame, draw_flag=True)# 
        
        # make sure the aruco's orientation in the camera view! 
        poses = estimate_markers_poses(corners, marker_size=0.03, intrinsics=intrinsics)  # Marker size in meters

        cv2.imshow('Camera', frame)
        # Exit on 'q' key press
        if cv2.waitKey(framedelay) & 0xFF == ord('q'):
            break

        # detected
        if ids is not None:
            for k, iden in enumerate(ids):
                iden = str(iden)
                if iden in poses_traj:
                    poses_traj[iden].append(list(poses[k]))
                else:
                    poses_traj[iden] = []

    
    with open(dst_path, "w") as json_file:
        json.dump(poses_traj, json_file, indent=4)
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    src_path="data/up_forward_cam.webm"
    src_path=0
    if src_path!=0:
        dst_path = src_path.replace('webm','json')
    else:
        dst_path = 'data/camera.json'
    print(src_path, dst_path)
    capture_poses(src_path=src_path, dst_path=dst_path)
