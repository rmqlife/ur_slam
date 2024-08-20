import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from utils.pose_util import *
import roboticstoolbox as rtb
import time

import pickle
def save_object(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        obj = pickle.load(file)
    return obj

def pose_to_T(pose):
    R, t = pose_to_Rt(pose)
    return Rt_to_T(R, t)

def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t.flatten()
    return T

def visualize(ur5, q, T_camera, T_slam):
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

class MyHandEye:
    def __init__(self):
        pass

    def save(self, filename):
        data = {
            'T_c2g': self.T_c2g,
            'T_t2c': self.T_t2c,
            'T_g2b': self.T_g2b,
            'joint_traj': self.joint_traj
        }
        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)
        self.T_c2g = data['T_c2g']
        self.T_t2c = data['T_t2c']
        self.T_g2b = data['T_g2b']
        self.joint_traj = data['joint_traj']

    def eye_in_hand(self, poses_c2t, poses_g2b):
        # target to camera, T_t2c -> inverse(slam_poses)
        self.T_t2c = []
        for p in poses_c2t:
            # Compute the inverse translation vector
            T = pose_to_T(p)
            self.T_t2c.append(np.linalg.inv(T))

        self.T_g2b = []
        for p in poses_g2b:
            T = pose_to_T(p) 
            self.T_g2b.append(T)
     
        R_gripper2base = [a[:3, :3] for a in self.T_g2b]
        t_gripper2base = [a[:3, 3] for a in self.T_g2b]
        R_target2cam = [b[:3, :3] for b in self.T_t2c]
        t_target2cam = [b[:3, 3] for b in self.T_t2c]

        import cv2
        R_c2g, t_c2g = cv2.calibrateHandEye(R_gripper2base=R_gripper2base, t_gripper2base=t_gripper2base, R_target2cam=R_target2cam, t_target2cam=t_target2cam)
        self.T_c2g = Rt_to_T(R_c2g, t_c2g)
        print("camera to gripper:",self.T_c2g)


        return self.T_c2g
        # # translate gripper pose to camera pose

    def compute_t2b(self):
        T_t2b= []
        for A,B in zip(self.T_g2b,self.T_t2c):
            T_t2b.append(A @ self.T_c2g @ B)
        return T_t2b

def validate_model(model_path):
    # test 
    myHandEye = MyHandEye()
    myHandEye.load(model_path)

    T_c2g, joint_traj = myHandEye.T_c2g, myHandEye.joint_traj
    T_t2b = myHandEye.compute_t2b()

    ur5 = rtb.models.UR5()
    T_camera = SE3(T_c2g)
    for i in range(len(joint_traj)):
        q = joint_traj[i]
        T_slam = SE3(T_t2b[i])
        T_slam.printline()
        # visualize(ur5, q, T_camera, T_slam)
        # plt.show()
        # input("press enter")
        # plt.clf()

def compute_model(data_dir):
    joints_traj = np.load(f'{data_dir}/traj.npy')

    slam_poses = np.load(f'{data_dir}/slam_poses.npy')
    robot_poses = np.load(f'{data_dir}/robot_poses.npy')

    myHandEye = MyHandEye()
    myHandEye.joint_traj = joints_traj
    myHandEye.eye_in_hand(poses_c2t=slam_poses, poses_g2b=robot_poses)
    myHandEye.save(f'{data_dir}/hand_eye.npz')
    return myHandEye


if __name__ == "__main__":
    data_dir = 'data/images-20240820-140928'

    myHandEye = compute_model(data_dir=data_dir)
    T_t2b = myHandEye.compute_t2b()
    T_b2t = SE3(T_t2b[0]).inv() # select one as T_t2b
    print(T_b2t)
    save_object(T_b2t, f'{data_dir}/base_transform.pkl')
    validate_model(model_path=f'{data_dir}/hand_eye.npz')
