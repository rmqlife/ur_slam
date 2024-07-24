from ik_step import *
from utils.pose_zoo import circle_pose

if __name__=="__main__":
    dry_run = True
    myIK = MyIK_rotate(base_transform)

    rospy.init_node('ik_step', anonymous=True)
    robot = init_real_robot()   
    joints = robot.get_joints()


    init_pose = myIK.fk(joints)
    target_pose = init_pose.copy()
    target_pose[2] -= 0.3

    points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)

    ax = visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=None)
    plt.show()

    traj = myIK.plan_trajectory(points, joints)
    myIK.show_traj(traj, loop=False)


    if not dry_run:
        for joints_star in traj:
            joints = robot.get_joints()
            # keep the joint 6 original position
            joints_star[5] = joints[5]
            joints_movement = np.max(np.abs(joints - joints_star))
            print(f"joints movement {joints_movement}")
            robot.move_joints(joints_star, duration=3*joints_movement, wait=True)
