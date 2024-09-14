import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from utils.pose_util import *
from utils.pose_zoo import *
import math
from spatialmath import SE3

class MyIK:
    def __init__(self):
        self.robot_show = rtb.models.UR5() #self.ur10()
        self.robot_plan = self.robot_show
            

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
        return SE3_to_pose(self.fk_se3(joints))
    
    def ik(self, pose, q):
        return self.ik_se3(pose_to_SE3(pose), q)
    
    def ik_point(self,point, q):
        '''
        given end effector position, find the joint configuration

        '''
        current_se3 = self.robot_plan.fkine(q)
        current_se3.t = [0,0,0]
        desired_pose = SE3(point) @ SE3(current_se3)

        return  self.robot_plan.ikine_LM(desired_pose, q0=q).q
        


    def fk_se3(self, joints):
        return self.robot_plan.fkine(joints)

    def ik_se3(self, se3, q):
        return self.robot_plan.ikine_LM(se3, q0=q).q
    
    def plan_trajectory(self, poses, joint_angles, vel_threshold=0.02):
        poses = np.array(poses)
        traj = None
        # if poses.shape[1]==3:
        #     # it is a points
        #     init_pose = self.fk(joint_angles)
        #     poses = append_vector(poses, init_pose[3:])

        print("poses!",poses)
        
        # end_joint_angles = joint_angles
        for i in range(len(poses)-1):
            start_pose = poses[i]
            end_pose = poses[i+1]
            # linear velocity 
            vel = np.linalg.norm(np.array(start_pose[:3])-np.array(end_pose[:3]))
            if vel<vel_threshold:
                num_steps=1
            else:
                print('vel threshold', vel_threshold)
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

    joints = [-1.57, -1.57, 1.57, -1.57, -1.57, 0]
    myIK = MyIK()
    pose = myIK.fk(joints)
    print('rtb', pose)

    ax = visualize_poses(pose, label='init pose', autoscale=False, ax=None)

    init_pose = pose.copy()
    if 1:
        target_pose = init_pose.copy()
        target_pose[2] -= 0.3
        points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
        # points = circle_points(init_pose[:3], radius=0.2, num_points=50)
        # points = rectangle_points(init_pose[:3], x=0.1, y=0.1)
        visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=ax)

        traj = myIK.plan_trajectory(points, joints)
        for j in traj:
            poses = myIK.fk(j)
        visualize_poses(poses, label="traj points", color='g', autoscale=False, ax=ax)
        plt.show()
        myIK.show_traj(traj,loop=True)

