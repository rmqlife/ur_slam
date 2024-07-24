
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
    glue = MyGlue(match_type="Aruco")

    id1=8
    id2=14

    home = '0612-facedown'

    image_path1 = f"{home}/rgb_{id1}.png"
    image_path2 = f"{home}/rgb_{id2}.png"
    traj_path = f"{home}/traj.npy"
    rgb1 = cv2.imread(image_path1)
    rgb2 = cv2.imread(image_path2)

    pts0, pts1 = glue.match(rgb1, rgb2)
    depth_path1 = replace_rgb_to_depth(image_path1)
    depth_path2 = replace_rgb_to_depth(image_path2)
    depth1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED)
    depth2 = cv2.imread(depth_path2, cv2.IMREAD_UNCHANGED)

    intrinsics = load_intrinsics("intrinsic_parameters.json")
    R, t = glue.match_3d(rgb1, rgb2, depth1, depth2, intrinsics)
    print(R, t)


    from archive.myIK_SLAM import MyIK_SLAM
    from pose_util import transform_pose
    ik_slam = MyIK_SLAM(slam_path='hand_eye_slam.npz', use_ikfast=True)

    joints_traj = np.load(traj_path)
    poses = ik_slam.forward_joints(joints_traj=joints_traj)

    pose1 = poses[id1]      
    pose2 = poses[id2]
    pose1_star= transform_pose(poses[id1], R, t)
    print(poses.shape)

    print_array(pose1)
    print_array(pose2)
    print_array(pose1_star)
