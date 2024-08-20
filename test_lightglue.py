
import cv2
import numpy as np
from utils.lightglue_util import MyGlue, replace_rgb_to_depth, load_intrinsics, print_array
import matplotlib.pyplot as plt

def plot_matching(image0, image1, pts0, pts1):
    """Plot matching keypoints between two images."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
    ax0.imshow(image0, cmap='gray')
    ax0.scatter(pts0[:, 0], pts0[:, 1], c='r', s=10)
    ax0.set_title('Image 1')
    ax0.axis('off')
    
    ax1.imshow(image1, cmap='gray')
    ax1.scatter(pts1[:, 0], pts1[:, 1], c='b', s=10)
    ax1.set_title('Image 2')
    ax1.axis('off')
    
    plt.show()

if __name__=="__main__":
    glue = MyGlue(match_type="LightGlue")

    id1=8
    id2=14

    home = 'slam_data/0613-slam-aruco'

    image_path1 = f"{home}/rgb_{id1}.png"
    image_path2 = f"{home}/rgb_{id2}.png"
    traj_path = f"{home}/traj.npy"
    rgb1 = cv2.imread(image_path1)
    rgb2 = cv2.imread(image_path2)
    print(rgb1.shape)
    pts0, pts1 = glue.match(rgb1, rgb2)
    print(pts0)
    show_frame = rgb1.copy()
    for (x1, y1), (x2, y2) in zip(pts0, pts1):
        cv2.line(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

    depth_path1 = replace_rgb_to_depth(image_path1)
    depth_path2 = replace_rgb_to_depth(image_path2)
    depth1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth_path2, cv2.IMREAD_UNCHANGED)

    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    R, t = glue.match_3d(pts0, pts1, depth1, depth2, intrinsics)
    print(R, t)
    cv2.imshow('show_frame', show_frame)
    # Exit on 'q' key press
    key = cv2.waitKey(0)
