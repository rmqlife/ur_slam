from ik_step import *


myIK = MyIK_rotate()

rospy.init_node('ik_step', anonymous=True)
robot = init_real_robot()   
init_joints = robot.get_joints()

init_pose = MyIK.fk
init_pose = pose.copy()
if 1:
    target_pose = init_pose.copy()
    target_pose[2] -= 0.3
    points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
    # points = circle_points(init_pose[:3], radius=0.2, num_points=50)
    # points = rectangle_points(init_pose[:3], x=0.1, y=0.1)
    visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=ax)

    traj = myIK.plan_trajectory(points, joint_angles)
    traj_poses = myIK.forward_joints(traj)
    visualize_poses(traj_poses, label="traj points", color='g', autoscale=False, ax=ax)
    plt.show()
    myIK.show_traj(traj,loop=True)
