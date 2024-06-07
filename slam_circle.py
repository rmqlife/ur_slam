from pose_zoo import circle_pose, rectangle_points
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


def append_vector(matrix, vector_to_append):
    assert len(matrix.shape) == 2, "Input matrix must be 2D"
    res = []
    for row in matrix:
        res.append(list(row)+list(vector_to_append))

    return np.array(res)


if __name__=="__main__":
    hand_eye_slam = load_object('./handeyeslam_data.pkl')
    real = True

    if real:
        import sys
        sys.path.insert(0,'/home/rmqlife/work/catkin_ur5/src/teleop/src')
        from myRobot import MyRobot
        import rospy
        rospy.init_node('ur5_slam', anonymous=True)
        robot=MyRobot()
        init_joints = robot.get_joints()
        
    else:
        joints_traj = np.load('robot_joints.npy')
        init_joints = joints_traj[0]
        init_joints[0]-=2*3.14

    # if not validate_joints(init_joints):
    #     exit()

    print("robot start at", init_joints)

    target_poses = circle_pose(center=[0,0.2,-0.1], toward=[0,0,-0.3], radius=0.1, num_points=20)
    
    rect_points = rectangle_points(center=[0, 0.2, -0.2], x=0.2, y=0.1)
    target_poses = append_vector(rect_points, [1,1,0,0])
    
    robot_poses = hand_eye_slam.slam_to_robot(target_poses)
    if False:
        visualize_poses(target_poses, 'target')
        visualize_poses(robot_poses, 'robot')
        plt.show()

    # remove orientation
    traj = plan_trajectory_points(robot_poses[:,:3], init_joints, num_steps=10)
    np.save('./rect_robot_traj.npy', traj)
    # show_traj(traj, loop=True)
    # exit()

    if real:
        duration=0.5

        from save_image import ImageSaver
        image_saver = ImageSaver()
        for i, joints in enumerate(traj[:]):
            print("moving to", joints)
            # keep joints-end
            robot.move_joints(joints, duration, wait=True)
            rospy.sleep(duration)
            image_saver.record()  # Save images
    else:
        show_traj(traj)
