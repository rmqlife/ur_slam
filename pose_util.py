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

def visualize_poses(poses, label="None", color='b', ax=None, autoscale=False):
    if ax is None:
        ax = plt.axes(projection ='3d')
        # Plot SLAM poses
        ax.set_autoscale_on(autoscale)


    poses = np.array(poses)
    if len(poses.shape)<2:
        poses = np.array([poses])

    # plot positions
    p = poses[:, :3]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=color, marker='o', label=label)

    if poses.shape[1]>3:
        # have orientation
        q = poses[:, 3:]
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
    ax.set_title("Poses with Quaternion")
    plt.legend()
    return ax

def R_dot_quat(R, quat):
    quat_mat = Rotation.from_quat(quat).as_matrix()
    quat_mat = np.dot(R, quat_mat)
    quat = Rotation.from_matrix(quat_mat).as_quat()
    return quat

def relative_rotation(q1, q2):
    # q1 to q2
    # Convert quaternions to rotation objects
    rot1 = Rotation.from_quat(q1)
    rot2 = Rotation.from_quat(q2)

    # Compute the relative rotation from rot1 to rot2
    relative_rotation = rot2 * rot1.inv()
    return relative_rotation.as_matrix()

def transform_pose(pose, R, t):
    pose_star = pose.copy()
    pose_star[:3] = np.dot(R, pose[:3])+t
    if len(pose)>3:
        pose_star[3:] =  R_dot_quat(R, pose[3:])
    return pose_star
 

def transform_poses(R, t, poses):
    # single vector
    if len(poses.shape)<2:
        poses = np.array([poses])
    transformed_poses = []
    for pose in poses:
        transformed_pose = transform_pose(pose, R, t)
        transformed_poses.append(transformed_pose)
    transformed_poses = np.vstack(transformed_poses)
    # single vector 
    if transformed_poses.shape[0]==1:
        transformed_poses = transformed_poses[0]
    return transformed_poses

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


def append_vector(matrix, vector_to_append):
    assert len(matrix.shape) == 2, "Input matrix must be 2D"
    res = []
    for row in matrix:
        res.append(list(row)+list(vector_to_append))

    return np.array(res)

def find_transformation(X, Y):
    """
    from X to Y
    Inputs: X, Y: lists of 3D points
    Outputs: R - 3x3 rotation matrix, t - 3-dim translation array.
    Find transformation given two sets of correspondences between 3D points.
    """
    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY

    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    return R, t