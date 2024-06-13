from pose_zoo import circle_pose, rectangle_points, circle_points
from pose_util import visualize_poses
import os
from hand_eye_slam import HandEyeSlam, load_object, poses_error
from myIK import MyIK
import numpy as np
import matplotlib.pyplot as plt

# make sure every joint is within -3.14 to +3.14, if not, +/- 6.28
def validate_joints(joints):
    for i, j in enumerate(joints):
        if abs(j)>3.14:
            print(f"joint {i} is over-rotated", joints)
            return False
    return True


class MyIK_SLAM(MyIK):
    def __init__(self, slam_path, use_ikfast=True):
        print(os.getcwd())
        self.slam = HandEyeSlam()
        self.slam.load()
        super().__init__(use_ikfast)


    def plan_trajectory(self, poses, joint_angles, vel_threshold=0.01):
        robot_poses = self.slam.slam_to_robot(poses, verbose=False)
        traj = super().plan_trajectory(robot_poses, joint_angles, vel_threshold)
        return traj

    def forward_joints(self, joints_traj):
        robot_poses = super().forward_joints(joints_traj)
        # print('robot_poses', robot_poses)
        return self.slam.robot_to_slam(robot_poses)



def dry_run(ik_slam, init_joints, target_poses):
    init_pose = ik_slam.forward_joints(init_joints)
    traj = ik_slam.plan_trajectory(target_poses, init_joints)

    # validate the traj
    print('traj', np.round(traj,2))
    poses = ik_slam.forward_joints(traj)

    ax = visualize_poses(init_pose, label='init pose', autoscale=False, ax=None)
    ax = visualize_poses(target_poses, label='target points', color='y', ax=ax)
    ax = visualize_poses(poses, label='poses', color='g', ax=ax)
    plt.show()

    ik_slam.show_traj(traj, loop=False)
    return traj
         
if __name__=="__main__":
    real = False
    ik_slam = MyIK_SLAM(slam_path='./hand_eye_slam.npz', use_ikfast=True)

    if real:
        import sys
        sys.path.insert(0,'/home/rmqlife/work/catkin_ur5/src/teleop/src')
        from myRobot import MyRobot
        import rospy
        rospy.init_node('ur5_slam', anonymous=True)
        robot=MyRobot()
        init_joints = robot.get_joints()
        
    else:
        # dry run with saved data
        joints_traj = np.load('slam_data/traj.npy')
        init_joints = joints_traj[10]
        init_joints[0]-=2*3.14
        print('init joints', init_joints)

    if not validate_joints(init_joints):
        exit()

    init_pose = ik_slam.forward_joints(init_joints)

    # circle = circle_pose(center=init_pose[:3], toward=init_pose[:3]+np.array([0,0,-0.3]), radius=0.1, num_points=10)
    circle = circle_points(center=init_pose[:3], radius=0.1, num_points=20)
    # rect =  rectangle_points(center=init_pose[:3], x=0.2, y=0.1)

    target = circle.copy()
    target.append(target[0])
    target.append(init_pose[:3])
    print(target)
    traj = dry_run(ik_slam=ik_slam, init_joints=init_joints, target_poses=target)

    if real:
        duration=0.2
        from myImageSaver import MyImageSaver
        image_saver = MyImageSaver()

        np.save(os.path.join(image_saver.folder_path,  'traj'), traj)
        for i, joints in enumerate(traj[:]):
            print("moving to", joints)
            # keep joints-end
            robot.move_joints(joints, duration, wait=True)
            image_saver.record()  # Save images
    
