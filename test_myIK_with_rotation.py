from ik_step import *
from utils.pose_zoo import circle_pose

myIK = MyIK_rotate()

rospy.init_node('ik_step', anonymous=True)
robot = init_real_robot()   
joints = robot.get_joints()

init_pose = MyIK.fk(joints)
init_pose = pose.copy()
if 1:
    target_pose = init_pose.copy()
    target_pose[2] -= 0.3
    points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)

    visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=ax)
    traj = myIK.plan_trajectory(points, joints)
    traj_poses = myIK.forward_joints(traj)
    visualize_poses(traj_poses, label="traj points", color='g', autoscale=False, ax=ax)
    plt.show()
    myIK.show_traj(traj, loop=True)
