import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from pose_util import *
from myIK import MyIK

def vec2mat(vec):
    if len(vec.shape)<2:
        vec = np.array([vec])   
    return vec

def poses_error(poses1, poses2):
    poses1 = vec2mat(poses1)
    poses2 = vec2mat(poses2)
    return np.mean(np.linalg.norm(poses1[:, :3] - poses2[:, :3], axis=1))

class HandEyeSlam:
    def __init__(self):
        pass

    def save(self, filename='hand_eye_slam.npz'):
        data = {
            'R': self.R,
            'R_inv': self.R_inv,
            't': self.t,
            'R_q': self.R_q,
            't_inv': self.t_inv,
            'R_q_inv': self.R_q_inv
        }
        np.savez(filename, **data)

    def load(self, filename='hand_eye_slam.npz'):
        data = np.load(filename)
        self.R = data['R']
        self.R_inv = data['R_inv']
        self.t = data['t']
        self.R_q = data['R_q']
        self.t_inv = data['t_inv']
        self.R_q_inv = data['R_q_inv']


    def estimate(self, slam_poses, robot_poses, verbose=False):
        if verbose:
            ax = visualize_poses(slam_poses, 'slam poses', color='b', ax=None)
            ax = visualize_poses(robot_poses, 'robot poses', color='g', ax=ax)
            plt.show()
        
        # save data to debug
        self.slam_poses = slam_poses
        self.robot_poses = robot_poses
        assert(slam_poses.shape[1]>=3)
        assert(slam_poses.shape==robot_poses.shape)



        self.R, self.t = icp(robot_poses[:, :3], slam_poses[:, :3])
        new_poses = transform_poses(self.R, self.t, slam_poses)
        if slam_poses.shape[1]==3:
            self.R_q = np.eye(3)
        else:
            self.R_q = relative_rotation(new_poses[0,3:], robot_poses[0,3:])

        # Compute the inverse translation vector
        self.R_inv = np.linalg.inv(self.R)
        self.t_inv = -np.dot(self.R_inv, self.t)

        self.R_q_inv = np.linalg.inv(self.R_q)
        pass

    def slam_to_robot(self, poses, verbose=False):
        poses = np.array(poses)
        # transform the point
        transformed_poses = transform_poses(self.R, self.t, poses) 
        # rotate based on the orientation difference between cam and robot
        rotated_poses = transform_poses(R=self.R_q, t=[0,0,0], poses=transformed_poses.copy())
        rotated_poses[:,:3] = transformed_poses[:, :3]
        if verbose:
            ax = visualize_poses(poses, 'slam poses', color='b')
            # ax = visualize_poses(transformed_poses, 'transformed', ax=ax)
            ax = visualize_poses(rotated_poses, 'rotated', color='g', ax=ax)
            plt.show()
        return rotated_poses
    
    def robot_to_slam(self, poses, verbose=False):
        poses = np.array(poses)
        transformed_poses = transform_poses(self.R_inv, self.t_inv, poses) 
        if verbose:
            ax = visualize_poses(poses, "robot poses", color='b')
            ax = visualize_poses(transformed_poses, "tranformed to slam poses", color='g', ax=ax)
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
    folder = 'slam_data'

    slam_poses = np.load(f'{folder}/slam_poses.npy')
    slam_poses[:, [6,3,4,5]] = slam_poses[:,[3,4,5,6]]

    joints_traj = np.load(f'{folder}/traj.npy')
    ik = MyIK()
    robot_poses = ik.forward_joints(joints_traj)

    print(robot_poses.shape, slam_poses.shape)

    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.estimate(slam_poses, robot_poses)
    # hand_eye_slam.load()

    new_poses = hand_eye_slam.slam_to_robot(slam_poses, verbose=1)
    print("overall projected error", poses_error(robot_poses, new_poses))

    new_poses = hand_eye_slam.robot_to_slam(robot_poses, verbose=1) 
    print("overall projected error", poses_error(slam_poses, new_poses))

    hand_eye_slam.save()