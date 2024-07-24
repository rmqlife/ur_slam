import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from utils.pose_util import *
import roboticstoolbox as rtb
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
            'T_t2c': self.T_t2c,
            'T_g2b': self.T_g2b,
            'joint_traj': self.joint_traj
        }
        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)
        self.R_c2g = data['R_c2g']
        self.t_c2g = data['t_c2g']
        self.T_t2c = data['T_t2c']
        self.T_g2b = data['T_g2b']
        self.joint_traj = data['joint_traj']

        self.R_g2c, self.t_g2c = inverse_Rt(self.R_c2g, self.t_c2g)

    def eye_in_hand(self, poses_c2t, poses_g2b):
        # target to camera, T_t2c -> inverse(slam_poses)
        T_t2c = []
        for p in poses_c2t:
            # Compute the inverse translation vector
            R, t = pose_to_Rt(p)
            matrix = np.eye(4)
            matrix[:3,:3] = R
            matrix[:3,3] = t
            T_t2c.append(np.linalg.inv(matrix))

        self.T_t2c = T_t2c
        T_g2b = []
        for p in poses_g2b:
            R, t = pose_to_Rt(p) 
            matrix = np.eye(4)
            matrix[:3,:3] = R
            matrix[:3,3] = t
            T_g2b.append(matrix)
        self.T_g2b = T_g2b
     
        R_gripper2base = [a[:3, :3] for a in T_g2b]
        t_gripper2base = [a[:3, 3] for a in T_g2b]
        R_target2cam = [b[:3, :3] for b in T_t2c]
        t_target2cam = [b[:3, 3] for b in T_t2c]

        import cv2
        R_c2g, t_c2g = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base, R_target2cam=R_target2cam, t_target2cam=t_target2cam)
        print("T_c2g", [R_c2g, t_c2g])
        # # translate gripper pose to camera pose
        self.R_g2c, self.t_g2c = inverse_Rt(R_c2g, t_c2g)
        self.R_c2g = R_c2g
        self.t_c2g = t_c2g

        # # visualize_poses(poses)


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
    # test 
    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.load(f'slam_data/0613-slam-aruco/hand_eye_slam_0724-1706.npz')

    R_c2g, t_c2g,T_t2c,T_g2b,joint_traj= hand_eye_slam.R_c2g, hand_eye_slam.t_c2g, hand_eye_slam.T_t2c, hand_eye_slam.T_g2b, hand_eye_slam.joint_traj
    T_c2g = np.eye(4)
    T_c2g[:3, :3] = R_c2g
    T_c2g[:3, 3] = t_c2g.flatten()  # Reshape t_cam2gripper to a 1D array
    T_t2b= []
    for A,B in zip(T_g2b,T_t2c):
        T_t2b.append(A @ T_c2g @ B)
    
    ur5 = rtb.models.UR5()
    matrix2 = np.eye(4)
    matrix2[:3,:3] = R_c2g
    matrix2[:3,3] = t_c2g.flatten()
    T_camera = SE3(matrix2)
    for i in range(len(joint_traj)):
        q = joint_traj[i]
        T_slam = SE3(T_t2b[i])
        # visualize(ur5, q, T_camera, T_slam)
        # plt.show()
        T_slam.printline()
        # plt.pause(0.1)
        # plt.clf()

def compute_model():
    folder = 'slam_data/0613-slam-aruco'
    joints_traj = np.load(f'{folder}/traj.npy')
    slam_poses = np.load(f'{folder}/slam_poses.npy')
    robot_poses = np.load(f'{folder}/robot_poses.npy')
    hand_eye_slam = HandEyeSlam()
    hand_eye_slam.joint_traj = joints_traj
    hand_eye_slam.eye_in_hand(poses_c2t=slam_poses, poses_g2b=robot_poses)
    hand_eye_slam.save(f'{folder}/hand_eye_slam_{time.strftime("%m%d-%H%M")}.npz')

def visualize(ur5, q,T_camera, T_slam):
    ur5.plot(q,backend='pyplot')
    T_ee = ur5.fkine(q)
    T_camera_base = T_ee * T_camera
    T_slam_base = T_slam
    # Plot the camera frame
    T_camera_base.plot(frame='C', color='red')
    T_slam_base.plot(frame='S', color='green')
    plt.gca().set_xlim([-1, 1])  # Set x-axis limits
    plt.gca().set_ylim([-1, 1])  # Set y-axis limits
    plt.gca().set_zlim([-1, 1])  # Set z-axis limits (if applicable)

if __name__ == "__main__":
    compute_model()
    validate_model()
