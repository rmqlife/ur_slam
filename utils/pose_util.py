import numpy as np
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spatialmath import SE3

unit_vector = [1,0,0]

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
            d = quat_to_R(q[i]) @ unit_vector
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

def SE3_to_pose(Rt):
    return list(Rt.t) + list(R_to_quat(Rt.R))

def pose_to_SE3(pose):
    se3 = SE3.Tx(0)
    se3.R[:] = quat_to_R(pose[3:])
    se3.t = pose[:3]
    return se3


def xyzw2wxyz(q):
    return [q[3], q[0], q[1], q[2]]

def wxyz2xyzw(q):
    return [q[1], q[2], q[3], q[0]]

def R_to_quat(R):
    import transforms3d.quaternions as tq
    # return wxyz2xyzw(tq.mat2quat(R))
    return Rotation.from_matrix(R).as_quat()

def quat_to_R(q):
    import transforms3d.quaternions as tq
    # return tq.quat2mat(xyzw2wxyz(q))
    return  Rotation.from_quat(q).as_matrix()

def pose_to_Rt(pose):
    R = quat_to_R(pose[3:])
    t = np.array(pose[:3])
    return R, t

def Rt_to_pose(R, t):
    pose = list(t) + list(R_to_quat(R))
    return np.array(pose)

def inverse_Rt(R, t):
    return R.T, -R.T @ t

def inverse_pose(pose):
    R, t = pose_to_Rt(pose)
    R_star, t_star = inverse_Rt(R, t)
    return Rt_to_pose(R_star, t_star)

def Rt_dot(R1, t1, R2, t2):
    #R2 = R1 dot R0
    R = np.dot(R1, R2)
    t = np.dot(R1, t2) + t1
    return R, t

def relative_rotation(q1, q2):
    # q1 to q2
    # Convert quaternions to rotation objects

    rot1 = quat_to_R(q1)
    rot2 = quat_to_R(q2)

    # Compute the relative rotation from rot1 to rot2
    relative_rotation = rot2 * rot1.inv()
    return relative_rotation.as_matrix()

def transform_pose(R, t, pose):
    pose_star = pose.copy()
    pose_star[:3] = np.dot(R, pose[:3])+t
    if len(pose)>3:
        pose_star[3:] =  R_to_quat(R @ quat_to_R(pose[3:]))
    return pose_star
 
def transform_poses(R, t, poses):
    # single vector
    poses = np.array(poses)
    if len(poses.shape)<2:
        return transform_pose(R, t, pose)
    transformed_poses = []
    for pose in poses:
        transformed_pose = transform_pose(R, t, pose)
        transformed_poses.append(transformed_pose)
    transformed_poses = np.vstack(transformed_poses)
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

def vec2mat(vec):
    if len(vec.shape)<2:
        vec = np.array([vec])   
    return vec

def poses_error(poses1, poses2):
    poses1 = vec2mat(poses1)
    poses2 = vec2mat(poses2)
    return np.mean(np.linalg.norm(poses1[:, :3] - poses2[:, :3], axis=1))

if __name__=="__main__":
    folder = 'slam_data/0613-slam-aruco'
    joints_traj = np.load(f'{folder}/traj.npy')
    slam_poses = np.load(f'{folder}/slam_poses.npy')
    robot_poses = np.load(f'{folder}/robot_poses.npy')

    visualize_poses(robot_poses)
    plt.show()