import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from utils.pose_util import *
from myIK import MyIK
import time

def Rt_dot(R1, t1, R2, t2):
    #R2 = R1 dot R0
    R = np.dot(R1, R2)
    t = np.dot(R1, t2) + t1
    return R, t

class HandEyeSlam:
    def __init__(self):
        pass

    def save(self, filename):
        data = {
            'R_c2g': self.R_c2g,
            't_c2g': self.t_c2g,
        }
        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)
        self.R_c2g = data['R_c2g']
        self.t_c2g = data['t_c2g']
        self.R_g2c, self.t_g2c = inverse_Rt(self.R_c2g, self.t_c2g)

    def eye_in_hand(self, poses_c2t, poses_g2b):
        # target to camera, T_t2c -> inverse(slam_poses)
        t_t2c = []
        R_t2c = []
        for p in poses_c2t:
            # Compute the inverse translation vector
            R, t = pose_to_Rt(p)
            R_inv, t_inv = inverse_Rt(R, t)
            t_t2c.append(t_inv)
            R_t2c.append(R_inv)

        t_g2b = []
        R_g2b = []
        for p in poses_g2b:
            R, t = pose_to_Rt(p)
            t_g2b.append(t)
            R_g2b.append(R)

        import cv2
        R_c2g, t_c2g = cv2.calibrateHandEye(R_gripper2base=R_g2b, t_gripper2base=t_g2b, R_target2cam=R_t2c, t_target2cam=t_t2c)
        t_c2g = t_c2g[:,0]
        print("T_c2g", [R_c2g, t_c2g])

        # translate gripper pose to camera pose
        self.R_g2c, self.t_g2c = inverse_Rt(R_c2g, t_c2g)
        self.R_c2g = R_c2g
        self.t_c2g = t_c2g

        # visualize_poses(poses)


    def slam_to_robot(self, poses, verbose=False):
        slam_poses = np.array(poses)
        # transform the point
        gripper_poses = transform_poses(self.R_c2g, self.t_c2g, slam_poses) 
        if verbose:
            ax = visualize_poses(slam_poses, 'slam poses', color='b')
            ax = visualize_poses(gripper_poses, 'transformed', color='g', ax=ax)
            plt.show()
        return gripper_poses
    
    def robot_to_slam(self, poses, verbose=False):
        
        gripper_poses = np.array(poses)
        transformed_poses = transform_poses(self.R_g2c, self.t_g2c, gripper_poses) 
        if verbose:
            ax = visualize_poses(gripper_poses, "gripper_poses", color='b')
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


def validate_model():
    folder = 'data/0613-slam-aruco'
    joints_traj = np.load(f'{folder}/traj.npy')
    slam_poses = np.load(f'{folder}/slam_poses.npy')
    ik = MyIK(use_ikfast=False)
    robot_poses = ik.forward_joints(joints_traj)

    print(robot_poses.shape, slam_poses.shape)

    # test 
    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.load(f'data/0613-slam-aruco/hand_eye_slam_0715-1649.npz')

    R_c2g, t_c2g = hand_eye_slam.R_c2g, hand_eye_slam.t_c2g

    # translate gripper pose to target
    # validate if t2b is fixed:
    poses = []
    for i in range(len(robot_poses)):
        R_g2b, t_g2b = pose_to_Rt(robot_poses[i])  
        R_c2t, t_c2t = pose_to_Rt(slam_poses[i])
        R_t2c, t_t2c = inverse_Rt(R_c2t, t_c2t)

        R_c2b, t_c2b = Rt_dot(np.eye(3), t_c2g, R_g2b, t_g2b)
        pose_c2b = Rt_to_pose(R_c2b, t_c2b)
        poses.append(pose_c2b)

        R_t2b_i, t_t2b_i = Rt_dot(R_t2c, t_t2c, R_c2b, t_c2b)
        print(i, t_t2b_i)

    ax = visualize_poses(poses, color='r',ax=None)
    visualize_poses(robot_poses, color='b', ax=ax)
    plt.show()
    # new_poses = hand_eye_slam.robot_to_slam(robot_poses, verbose=1) 
    # print("overall projected error", poses_error(slam_poses, new_poses))
    pass

def compute_model():
    folder = 'data/0613-slam-aruco'
    joints_traj = np.load(f'{folder}/traj.npy')
    slam_poses = np.load(f'{folder}/slam_poses.npy')

    ik = MyIK(False)
    robot_poses = ik.forward_joints(joints_traj)
    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.eye_in_hand(poses_c2t=slam_poses, poses_g2b=robot_poses)
    hand_eye_slam.save(f'{folder}/hand_eye_slam_{time.strftime("%m%d-%H%M")}.npz')

if __name__ == "__main__":
    validate_model()
    # compute_model()
    pass