import rospy
import numpy as np
from myRTABMap import RTABMapPoseListener

rospy.init_node('ur5_shake_test', anonymous=True)

traj = np.load('./images-20240611-201439/traj.npy' )

import sys
sys.path.insert(0,'/home/rmqlife/work/catkin_ur5/src/teleop/src')
from myRobot import MyRobot
robot =  MyRobot()
print(traj)

duration = 0.3
slam_pose_listener = RTABMapPoseListener(verbose=False)
slam_poses = []

for i, joints in enumerate(traj[:]):
    print("moving to", joints)

    # keep joints-end
    # joints[-1]=0
    robot.move_joints(joints, duration, wait=True)
    rospy.sleep(duration)
    print('now at', robot.get_joints()) 

    # read RTABMap
    slam_position = slam_pose_listener.get_position()
    slam_orientation = slam_pose_listener.get_orientation()
    print("slam position", slam_position)
    print("slam orientation", slam_orientation)

    # Save slam pose
    slam_pose = np.array(list(slam_position) + list(slam_orientation))
    slam_poses.append(slam_pose)

# Convert to NumPy array and save
slam_poses = np.vstack(slam_poses)
np.save('slam_poses.npy', slam_poses)
