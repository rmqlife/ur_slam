from ik_step import init_robot, MyIK_rotate, MyRobot_with_IK
from utils.pose_zoo import circle_pose
from hand_eye_calib import *
import rospy

base_transform = SE3.Rx(135, unit='deg')

if __name__=="__main__":

    rospy.init_node('ik_step', anonymous=True)
    dry_run = False
    base_transform = load_object("slam_data/base_transform.pkl")
    print("base transform", base_transform)
    
    robot  =  init_robot() 
    joints = robot.get_joints()
    print(joints)
    init_pose = robot.get_pose()
    target_pose = init_pose.copy()
    target_pose[2] -= 0.3

    points = circle_pose(init_pose, target_pose[:3], radius=0.1, num_points=50)
    ax = visualize_poses(points, label="points to plan", color='y', autoscale=False, ax=None)
    plt.show()

    traj = robot.myIK.plan_trajectory(points, joints)
    robot.myIK.show_traj(traj, loop=dry_run)

    if not dry_run:
        for joints_star in traj:
            joints = robot.get_joints()
            # keep the joint 6 original position
            joints_star[5] = joints[5]
            joints_movement = np.max(np.abs(joints - joints_star))
            print(f"joints movement {joints_movement}")
            robot.move_joints(joints_star, duration=3*joints_movement, wait=True)
