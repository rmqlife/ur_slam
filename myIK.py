import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from ur_ikfast import ur_kinematics
import roboticstoolbox as rtb
from pose_util import visualize_poses, append_vector
import math

class MyIK:
    def __init__(self, use_ikfast=True):
        self.robot_show = rtb.models.UR5()  
        if use_ikfast:
            self.robot_plan = ur_kinematics.URKinematics('ur5e')
        else:
            self.robot_plan = self.robot_show
            
        self.use_ikfast=use_ikfast

    def forward_joints(self, joints_traj):
        joints_traj = np.array(joints_traj)
        if len(joints_traj.shape) < 2:
            joints_traj = np.array([joints_traj])
        
        poses = []
        for joints in joints_traj:
            if self.use_ikfast:
                pose = self.robot_plan.forward(joints)
            else:
                Rt = self.robot_plan.fkine(joints)
                pose = list(Rt.t) + list(self.R_to_quaternion(Rt.R))
            poses.append(pose)

        poses = np.array(poses)
        if poses.shape[0] < 2:
            poses = poses[0]
        return poses

    def R_to_quaternion(self, R):
        rotation = Rotation.from_matrix(R)
        quaternion = rotation.as_quat()
        return quaternion


    def plan_trajectory(self, poses, joint_angles, vel_threshold=0.02):
        poses = np.array(poses)
        traj = None
        if poses.shape[1]==3:
            # it is a points
            init_pose = self.robot_plan.forward(joint_angles)
            poses = append_vector(poses, init_pose[3:])

        for i in range(len(poses)-1):
            start_pose = poses[i]
            end_pose = poses[i+1]
            # linear velocity 
            vel = np.linalg.norm(np.array(start_pose[:3])-np.array(end_pose[:3]))
            if vel<vel_threshold:
                num_steps=1
            else:
                num_steps=int(math.floor(vel/vel_threshold))
            print(num_steps)
            
            start_joint_angles = self.robot_plan.inverse(start_pose, all_solutions=False, q_guess=joint_angles)
            end_joint_angles = self.robot_plan.inverse(end_pose, all_solutions=False, q_guess=joint_angles)
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

    joint_angles = [-3.14, -1.57, 1.57, -1.57, -1.57, 0]
    myIK = MyIK(use_ikfast=False)
    pose = myIK.forward_joints(joint_angles)
    print('rtb', pose)

    myIK = MyIK(use_ikfast=True)
    pose = myIK.forward_joints(joint_angles)
    print('ikfast', pose)

    ax = visualize_poses(pose, label='init pose', autoscale=False, ax=None)

    init_pose = pose.copy()
    if 1:
        from pose_zoo import circle_pose, circle_points

        # target_pose = init_pose.copy()
        # target_pose[2] -= 0.3
        # poses = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
        # visualize_poses(poses, label="circle points", autoscale=True)
        
        # traj = myIK.plan_trajectory(poses, joint_angles, num_steps=5)
        # myIK.show_traj(traj)
        

        points = circle_points(init_pose[:3], radius=0.1, num_points=10)
        points.append(points[0])
        points.append(init_pose[:3])
        visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=ax)

        traj = myIK.plan_trajectory(points, joint_angles)
        
        traj_poses = myIK.forward_joints(traj)
        visualize_poses(traj_poses, label="traj points", color='g', autoscale=False, ax=ax)
        plt.show()
        myIK.show_traj(traj,loop=True)
