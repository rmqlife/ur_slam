import cv2
import numpy as np
# Assuming lightglue_util is a module you've created or imported from somewhere
from utils.lightglue_util import MyGlue, load_intrinsics, print_array
from myIK_SLAM import MyIK_SLAM, poses_error
from utils.pose_util import transform_pose

home = 'data/0612-facedown'
id_src = 8
id_dst = 26
intrinsics_path="slam_data/intrinsics_d435.json"
slam_path = 'data/0613-slam/hand_eye_slam.npz'

def load_rgb_depth(img_id):
    # Fixed the incorrect variable name depth_path2 to depth_path
    image_path = f"{home}/rgb_{img_id}.png"
    depth_path = f"{home}/depth_{img_id}.png"  # Assuming depth images have a different naming convention
    rgb = cv2.imread(image_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    return rgb, depth    

if __name__ == "__main__":
    glue = MyGlue(match_type="LightGlue") # Aruco LightGlue

    # Fixed the variable names to match the function definition
    src_rgb, src_depth = load_rgb_depth(id_src)
    dst_rgb, dst_depth = load_rgb_depth(id_dst)  # Assuming you want to load the second image as the destination

    traj_path = f"{home}/traj.npy"
    intrinsics = load_intrinsics(intrinsics_path)
    # Fixed the variable names to match the loaded images
    src_pts, dst_pts = glue.match(src_rgb, dst_rgb)
    R, t = glue.match_3d(src_pts, dst_pts, src_depth, dst_depth, intrinsics, show=False)
    print("transformation", R, t)

    ik = MyIK_SLAM(slam_path=slam_path, use_ikfast=False)

    joints_traj = np.load(traj_path)
    poses = ik.forward_joints(joints_traj=joints_traj)

    pose_src = poses[id_src]  # Renamed pose1 to pose_src
    pose_dst = poses[id_dst]   # Renamed pose2 to pose_dst for consistency
    pose_dst_star = transform_pose(R, t, pose_src)  # Using the new variable name
    print(poses.shape)

    print_array(pose_src)
    print_array(pose_dst)
    print_array(pose_dst_star)
    print(poses_error(pose_dst, pose_dst_star))
    # R, t