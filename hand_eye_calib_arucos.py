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


class MyHandEye:
    def __init__(self):
        pass

    def save(self, filename):
        data = {
            'T_c2g': self.T_c2g,
            'T_t2c': self.T_t2c,
            'T_g2b': self.T_g2b,
        }
        np.savez(filename, **data)

    def load(self, filename):
        data = np.load(filename)
        self.T_c2g = data['T_c2g']
        self.T_t2c = data['T_t2c']
        self.T_g2b = data['T_g2b']

    def eye_in_hand(self, poses_m2c, poses_g2b):
        # maker 2 camera -> T_t2c -
        self.T_t2c = []
        for p in poses_m2c:
            T = pose_to_T(p)
            self.T_t2c.append(T)

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
    

def compute_model(folder):

    maker_poses = np.load(f'{folder}/marker_poses.npy')
    robot_poses = np.load(f'{folder}/robot_poses.npy')
    myHandEye = MyHandEye()
    myHandEye.eye_in_hand(poses_m2c=maker_poses, poses_g2b=robot_poses)
    myHandEye.save(f'{folder}/hand_eye.npz')
    return myHandEye

def validate_model(model_path):
    # test 
    myHandEye = MyHandEye()
    myHandEye.load(model_path)
    T_c2g = myHandEye.T_c2g
    T_t2b = myHandEye.compute_t2b()
    for i in range(len(T_t2b)):
        T_slam = SE3(T_t2b[i])
        T_slam.printline()

if __name__ == "__main__":
    data_dir = 'data/images-20240822-141709'
    myHandEye = compute_model(data_dir)
    
    T_t2b = myHandEye.compute_t2b()
    print(SE3(myHandEye.T_c2g))
    for i in range(len(T_t2b)):
        T_b2t = SE3(T_t2b[i]).inv() # select one as T_t2b
        T_b2t.printline()
        
    # print(T_b2t)
    print("\\\\\\\\")
    # save_object(T_b2t, f'{data_dir}/base_transform.pkl')
    # validate_model(model_path=f'{data_dir}/hand_eye.npz')