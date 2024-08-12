import cv2
import cv2.aruco as aruco
import numpy as np
from utils.pose_util import *
from utils.lightglue_util import load_intrinsics
import json

ARUCO_DICT_NAME = aruco.DICT_4X4_50
my_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)

# Function to detect ArUco markers
def detect_aruco(image, draw_flag=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(my_aruco_dict, aruco.DetectorParameters())
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

    if corners is  None or ids is None:
        return corners, ids
    else:
        # Draw detected markers on the image
        if draw_flag and ids is not None:
            image = aruco.drawDetectedMarkers(image, corners, ids)
        corners = [c[0] for c in corners]
        ids = [c[0] for c in ids]
    return corners, ids


# Function to generate ArUco markers
def generate_aruco_marker(marker_id, marker_size, output_file):
    # Generate ArUco marker image
    marker_image = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker_image = aruco.generateImageMarker(my_aruco_dict, marker_id, marker_size, marker_image, 1)
    cv2.imwrite(output_file, marker_image)


def estimate_markers_poses(corners, marker_size, intrinsics):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    '''
    # make sure the aruco's orientation in the camera view! 
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    mtx = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                    [0, intrinsics["fy"], intrinsics["cy"]],
                    [0, 0, 1]], dtype=np.float32)
    distortion = np.zeros((5, 1))  # Assuming no distortion

    poses = []
    for c in corners:
        ret, rvec, tvec = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_ITERATIVE)
        # visualize
        # print('pnp res', (ret, rvec, tvec))
        tvec = tvec.reshape((3))
        R = cv2.Rodrigues(rvec)[0]
        pose = Rt_to_pose(R, tvec)
        poses.append(pose)
    return poses


    