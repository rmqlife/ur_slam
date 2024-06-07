import numpy as np
import math

def plan_trajectory(poses,  joint_angles, num_steps):
    from ur_ikfast import ur_kinematics
    import roboticstoolbox as rtb
    traj = None
    robot_plan = ur_kinematics.URKinematics('ur5e')
    end_joint_angles = joint_angles
    for i in range(len(poses)-1):
        start_pose = poses[i]
        end_pose = poses[i+1]
        start_joint_angles = robot_plan.inverse(start_pose, all_solutions=False, q_guess=joint_angles)
        end_joint_angles = robot_plan.inverse(end_pose, all_solutions=False, q_guess=joint_angles)
        traj_segment = rtb.jtraj(start_joint_angles, end_joint_angles, num_steps)
        if traj is None:
            traj = traj_segment.q
        else:
            traj = np.concatenate((traj, traj_segment.q), axis=0)
    return traj

def plan_trajectory_points(points,  joint_angles, num_steps):
    from ur_ikfast import ur_kinematics
    robot_plan = ur_kinematics.URKinematics('ur5e')
    init_pose = robot_plan.forward(joint_angles)
    poses = []
    for point in points:
        pose = list(point) + list(init_pose[3:])
        poses.append(pose)
    return plan_trajectory(poses, joint_angles, num_steps)

def show_traj(traj, loop=False):
    import roboticstoolbox as rtb
    robot_show = rtb.models.UR5()  # Create UR5 robot model
    robot_show.plot(traj, backend='pyplot',loop=loop)

def forward_joints(joints_traj):
    from ur_ikfast import ur_kinematics
    robot_plan = ur_kinematics.URKinematics('ur5e')
    poses = []
    for joints in joints_traj:
        pose = robot_plan.forward(joints)
        poses.append(pose)
    return np.vstack(poses)


if __name__=="__main__":
    from ur_ikfast import ur_kinematics
    robot_plan = ur_kinematics.URKinematics('ur5e')

    joint_angles = [-3.14,-1.57,1.57,-1.57,-1.57,0]

    init_pose = robot_plan.forward(joint_angles)
    # poses = rectangle_poses(init_pose)
    target_pose = init_pose.copy()
    target_pose[2] -= 0.3

    from pose_zoo import circle_pose
    from pose_util import visualize_poses
    import matplotlib.pyplot as plt
    poses = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
    # visualize_poses(poses)
    # plt.show()
    print(init_pose[3:])
    traj = plan_trajectory(poses, joint_angles, num_steps=5)
    show_traj(traj)



