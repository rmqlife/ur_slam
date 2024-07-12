from utils.aruco_util import *


def main():
    # Load camera calibration parameters (fx, fy, cx2, cy, distortion coefficients)
    cap = cv2.VideoCapture("utils/arucos.webm")  # Change to your camera index or video file path
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    framedelay=1000//30
    # framedelay=1
    poses_traj = dict()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Estimate camera pose using ArUco markers
        corners, ids = detect_aruco(frame, draw_flag=True)# 
        
        poses = estimate_markers_poses(corners, marker_size=0.03, intrinsics=intrinsics)  # Marker size in meters
        
        # print([corners[0].shape, ids[0]])
        cv2.imshow('Camera', frame)
        # Exit on 'q' key press
        if cv2.waitKey(framedelay) & 0xFF == ord('q'):
            break
        



main()