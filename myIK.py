import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import roboticstoolbox as rtb
from utils.pose_util import *
from utils.pose_zoo import circle_pose, circle_points, rectangle_points
import math
from spatialmath import SE3

class MyIK:
    def __init__(self, use_ikfast=False):
        self.robot_show = rtb.models.UR5() #self.ur10()
        if use_ikfast:
            from ur_ikfast import ur_kinematics
            self.robot_plan = ur_kinematics.URKinematics('ur5e')
        else:
            self.robot_plan = self.robot_show
            
        self.use_ikfast=use_ikfast

    def ur10(self):
        ## UR10 DH Parameters
        L1 = rtb.DHLink(d=0.1273,  a=0,     alpha=np.pi/2, offset=0)
        L2 = rtb.DHLink(d=0,       a=-0.612, alpha=0,     offset=0)
        L3 = rtb.DHLink(d=0,       a=-0.5723, alpha=0,    offset=0)
        L4 = rtb.DHLink(d=0.163941, a=0,    alpha=np.pi/2, offset=0)
        L5 = rtb.DHLink(d=0.1157,   a=0,    alpha=-np.pi/2, offset=0)
        L6 = rtb.DHLink(d=0.0922,  a=0,    alpha=0,     offset=0)

        ## Joint limits
        L1.qlim = [-2*np.pi, 2*np.pi]
        L2.qlim = [-2*np.pi, 2*np.pi]
        L3.qlim = [-2*np.pi, 2*np.pi]
        L4.qlim = [-2*np.pi, 2*np.pi]
        L5.qlim = [-2*np.pi, 2*np.pi]
        L6.qlim = [-2*np.pi, 2*np.pi]
        return rtb.DHRobot([L1, L2, L3, L4, L5, L6], name='UR10')

    def fk(self, joints):
        if self.use_ikfast:
            pose = self.robot_plan.forward(joints)
        else:
            Rt = self.robot_plan.fkine(joints)
            pose = SE3_to_pose(Rt)
        return pose
    
    def ik(self, pose, q):
        if self.use_ikfast:
            joints = self.robot_plan.inverse(pose, all_solutions=False, q_guess=q)
        else:
            # t = rtb
            Rt = pose_to_SE3(pose)
            joints = self.robot_plan.ikine_LM(Rt, q0=q).q
        return joints

    def forward_joints(self, joints_traj):
        joints_traj = np.array(joints_traj)
        if len(joints_traj.shape) < 2:
            joints_traj = np.array([joints_traj])
        
        poses = []
        for joints in joints_traj:
            pose = self.fk(joints)
            poses.append(pose)

        poses = np.array(poses)
        if poses.shape[0] < 2:
            poses = poses[0]
        return poses

    def plan_trajectory(self, poses, joint_angles, vel_threshold=0.02):
        poses = np.array(poses)
        traj = None
        if poses.shape[1]==3:
            # it is a points
            init_pose = self.forward_joints(joint_angles)
            poses = append_vector(poses, init_pose[3:])
        
        # end_joint_angles = joint_angles
        for i in range(len(poses)-1):
            start_pose = poses[i]
            end_pose = poses[i+1]
            # linear velocity 
            vel = np.linalg.norm(np.array(start_pose[:3])-np.array(end_pose[:3]))
            if vel<vel_threshold:
                num_steps=1
            else:
                num_steps=int(math.floor(vel/vel_threshold))
            
            start_joint_angles = self.ik(start_pose, joint_angles)
        
            end_joint_angles = self.ik(end_pose, joint_angles)
            traj_segment = rtb.jtraj(start_joint_angles, end_joint_angles, num_steps)
            if traj is None:
                traj = traj_segment.q
            else:
                traj = np.concatenate((traj, traj_segment.q), axis=0)
        return traj


    def show_traj(self, traj, loop=True):
        self.robot_show.plot(traj, backend='pyplot', loop=loop)

    def move_pose_by_Rt(self, pose, R, t):
        pass
    
if __name__ == "__main__":

    joint_angles = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
    myIK = MyIK(use_ikfast=False)
    pose = myIK.forward_joints(joint_angles)
    print('rtb', pose)

    # myIK = MyIK(use_ikfast=True)
    # pose = myIK.forward_joints(joint_angles)
    # print('ikfast', pose)

    ax = visualize_poses(pose, label='init pose', autoscale=False, ax=None)

    init_pose = pose.copy()
    if 1:
        target_pose = init_pose.copy()
        target_pose[2] -= 0.3
        # points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
        points = circle_points(init_pose[:3], radius=0.2, num_points=50)
        # points = rectangle_points(init_pose[:3], x=0.1, y=0.1)
        visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=ax)

        traj = myIK.plan_trajectory(points, joint_angles)
        traj_poses = myIK.forward_joints(traj)
        visualize_poses(traj_poses, label="traj points", color='g', autoscale=False, ax=ax)
        plt.show()
        myIK.show_traj(traj,loop=True)
