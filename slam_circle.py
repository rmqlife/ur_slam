from pose_zoo import circle_pose
from pose_util import visualize_poses

from hand_eye_slam import HandEyeSlam, load_object
from myIK import plan_trajectory_points, show_traj
import numpy as np
import matplotlib.pyplot as plt

# make sure every joint is within -3.14 to +3.14, if not, +/- 6.28
def validate_joints(joints):
    for i, j in enumerate(joints):
        if abs(j)>3.14:
            print(f"joint {i} is over-rotated", joints)
            return False
    return True


if __name__=="__main__":
    verbose = False
    hand_eye_slam = load_object('./handeyeslam_data.pkl')

    joints_traj = np.load('robot_joints.npy')
    init_joints = joints_traj[0]
    init_joints[0]-=2*3.14
    if validate_joints(init_joints):
        print("robot start at", init_joints)

        circle_poses = circle_pose(center=[0,0,0], toward=[0,0,-0.3], radius=0.1, num_points=15)

        robot_poses = hand_eye_slam.slam_to_robot(circle_poses)
        if verbose:
            visualize_poses(circle_poses, 'circle')
            visualize_poses(robot_poses, 'circle')
            plt.show()

        # remove orientation
        traj = plan_trajectory_points(robot_poses[:,:3], init_joints, num_steps=5)
        show_traj(traj)
    

    