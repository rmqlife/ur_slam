import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def normalize_vector(vector, l):
    # Calculate the length of the vector
    vector_length = np.linalg.norm(vector)
    # Normalize the vector
    normalized_vector = (vector / vector_length) * l
    return normalized_vector

def visualize_poses(poses, title="Poses with Quaternion"):
    # Extract positions and orientations
    p = poses[:, :3]
    q = poses[:, 3:]
    # Create a 3D plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = plt.axes(projection ='3d')
    # Plot SLAM poses
    ax.set_autoscale_on(False)

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c='b', marker='o', label='SLAM Poses')
    # Plot orientation vectors
    for i in range(len(p)):
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(q[i]).as_matrix()
        d = np.dot(R, [0,0,1])
        d = normalize_vector(d, l=0.2)
        ax.quiver(p[i, 0], p[i, 1], p[i, 2], 
                  d[0], d[1], d[2], 
                  length=0.1, normalize=True, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.legend()


def R_dot_quat(R, quat):
    quat_mat = Rotation.from_quat(quat).as_matrix()
    quat_mat = np.dot(R, quat_mat)
    quat = Rotation.from_matrix(quat_mat).as_quat()
    return quat


def transform_pose(pose, R, t):
    pose_star = pose.copy()
    pose_star[:3] = np.dot(R, pose[:3])+t
    pose_star[3:] =  R_dot_quat(R, pose[3:])
    return pose_star
 

def transform_poses(R, t, poses):
    transformed_poses = []
    for pose in poses:
        transformed_pose = transform_pose(pose, R, t)
        transformed_poses.append(transformed_pose)
    return np.vstack(transformed_poses)

def icp(poses1, poses2, max_iter=10, threshold=1e-5):
    # Initial guess for transformation
    R_est = np.eye(3)
    t_est = np.zeros(3)
    for _ in range(max_iter):
        # Estimate transformation
        R_est_new, t_est_new = estimate_transform(poses1, poses2, R_est, t_est)
        print("icp err", np.linalg.norm(t_est_new - t_est))
        # # Check convergence
        if np.linalg.norm(R_est_new - R_est) < threshold and np.linalg.norm(t_est_new - t_est) < threshold:
            break
        # Update transformation
        R_est = R_est_new.copy()
        t_est = t_est_new.copy()
    return R_est, t_est

def estimate_transform(poses1, poses2, R_init, t_init):
    # Compute centroids
    centroid1 = np.mean(poses1, axis=0)
    centroid2 = np.mean(poses2, axis=0)
    # Compute centered poses
    centered_poses1 = poses1 - centroid1
    centered_poses2 = poses2 - centroid2

    from scipy.linalg import orthogonal_procrustes
    R_est, _ = orthogonal_procrustes(centered_poses1, centered_poses2)
    # Compute translation
    t_est = centroid1 - np.dot(R_est, centroid2)
    return R_est, t_est


if __name__=="__main__":

    if False:
        R = [[ 0.97292059,-0.11727709, 0.19917731],
        [ 0.15271534, 0.97299991,-0.17305837],
        [-0.17350372, 0.19878948, 0.96456166]]
        
        t = [-0.09, 0.08,-0.01]

        pose1, pose2 = load_poses(50,105)
        pose3 = apply_Rt(pose1, R, t)
        
        compare_pose(pose1, pose2, pose3)
    else:
        traj = np.load('./show_traj.npy')
