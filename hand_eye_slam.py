import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from pose_util import *
from quaternion_util import relative_rotation

def poses_error(poses1, poses2):
    return np.mean(np.linalg.norm(poses1[:, :3] - poses2[:, :3], axis=1))

class HandEyeSlam:
    def __init__(self, slam_poses, robot_poses):
        self.estimate(slam_poses, robot_poses)

    def estimate(self, slam_poses, robot_poses):
        self.R, self.t = icp(robot_poses[:, :3], slam_poses[:, :3])
        new_poses = transform_poses(self.R, self.t, slam_poses)
        self.R_q = relative_rotation(new_poses[0,3:], robot_poses[0,3:])

        # Compute the inverse translation vector
        self.R_inv = np.linalg.inv(self.R)
        self.t_inv = -np.dot(self.R_inv, self.t)
        pass

    def slam_to_robot(self, poses, verbose=False):
        # transform the point
        transformed_poses = transform_poses(self.R, self.t, poses) 
        # rotate based on the orientation difference between cam and robot
        rotated_poses = transform_poses(R=self.R_q, t=[0,0,0], poses=transformed_poses)
        rotated_poses[:,:3] = transformed_poses[:, :3]
        if verbose:
            visualize_poses(poses, 'slam poses')
            visualize_poses(transformed_poses, 'transformed')
            visualize_poses(rotated_poses, 'rotated')
            plt.show()
        return rotated_poses
    
    def robot_to_slam(self, poses, verbose=False):
        transformed_poses = transform_poses(self.R_inv, self.t_inv, poses) 
        if verbose:
            visualize_poses(poses)
            visualize_poses(transformed_poses)
            plt.show()
        return transformed_poses


import pickle
def save_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj


if __name__ == "__main__":

    slam_poses = np.load('slam_poses.npy')
    print(slam_poses.shape)

    joints_traj = np.load('robot_joints.npy')
    from myIK import forward_joints
    robot_poses = forward_joints(joints_traj)
    print(robot_poses.shape)

    hand_eye_slam = HandEyeSlam(slam_poses=slam_poses, robot_poses=robot_poses)
    new_poses = hand_eye_slam.slam_to_robot(slam_poses, verbose=False)
    print("overall projected error", poses_error(robot_poses, new_poses))

    new_poses = hand_eye_slam.robot_to_slam(robot_poses, verbose=False) 
    print("overall projected error", poses_error(slam_poses, new_poses))

    save_object(hand_eye_slam,'handeyeslam_data.pkl')