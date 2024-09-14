import numpy as np
import roboticstoolbox as rtb
from utils.pose_util import *
from spatialmath import SE3, SO3
from myIK import MyIK
import rospy
from std_msgs.msg import Float32
import cv2
from follow_aruco import *
from ros_utils.myRobotNs import MyRobotNs
from ros_utils.myGripper import MyGripper
intrinsics = load_intrinsics("/home/rmqlife/work/ur_slam/slam_data/intrinsics_d435.json")

key_map = {
    ord('!'): -1,  # Shift+1
    ord('@'): -2,  # Shift+2
    ord('#'): -3,  # Shift+3
    ord('$'): -4,  # Shift+4
    ord('%'): -5,  # Shift+5
    ord('^'): -6,  # Shift+6
    ord('1'): 1,   # 1
    ord('2'): 2,   # 2
    ord('3'): 3,   # 3
    ord('4'): 4,   # 4
    ord('5'): 5,   # 5
    ord('6'): 6    # 6
}


def lookup_action(code, t_move=0.02, r_move=5):
    if abs(code)<=3:
        movement = t_move * np.sign(code)
    else:
        movement = r_move * np.sign(code)
    if abs(code)==1:
        return SE3.Tx(movement)
    elif abs(code)==2:
        return SE3.Ty(movement)
    elif abs(code)==3:
        return SE3.Tz(movement)
    elif abs(code)==4:
        return SE3.Rx(movement, unit='deg')
    elif abs(code)==5:
        return SE3.Ry(movement, unit='deg')
    elif abs(code)==6:
        return SE3.Rz(movement, unit='deg')
    return None


class MyIK_rotate(MyIK):
    def __init__(self, transform):
        self.transform = transform
        super().__init__()

    def fk_se3(self, joints):
        pose_se3 = super().fk_se3(joints)
        return self.transform * pose_se3

    def ik_se3(self, pose, q):
        pose = self.transform.inv() * pose
        return super().ik_se3(pose, q)
    
    def ik_point(self,point,q):
        point_robot = self.transform.inv() @ point
        return super().ik_point(point_robot.t,q)



class MySubscriber:
    def __init__(self, topic_name, topic_type):
        rospy.Subscriber(topic_name, topic_type, self.callback)
        # self.start_listening()
        self.get_return = False
        while not self.get_return:
            print('waiting for topic to publish...')
            rospy.sleep(0.5)

    def callback(self, data):
        # rospy.loginfo(f"Received float value: {data.data}")
        self.data = data.data
        self.get_return = True

    def start_listening(self):
        rospy.spin()

class MyRobot_with_IK(MyRobotNs):
    def __init__(self,myIK,ns):
        self.myIK = myIK
        self.shouldMove = False
        super().__init__(ns)

    def step(self, action, wait):
        pose_se3 = pose_to_SE3(self.get_pose())
        print('action print'), action.printline()
        pose_se3_new = action * pose_se3
        # if np.linalg.norm(action.t)<0.001:
        #     # rotation keep the x, y, z
        pose_se3_new.t = pose_se3.t + action.t
        return self.goto_pose(pose_se3_new, wait)

    def get_pose(self):
        return self.myIK.fk(super().get_joints())
    
    def goto_pose(self, pose, wait, coef=3, joint_thresh=1):
        joints = super().get_joints()
        pose_now = self.myIK.fk_se3(joints)
        joints_star = self.myIK.ik_se3(pose, q=joints)
        # compute the difference between joints and joints_star
        joints_movement = np.max(np.abs(joints - joints_star))
        print(f"joints movement {joints_movement}")
        if joints_movement>joint_thresh:
            print('something wrong with goto_pose()')
            pose = pose_now
        else:
            super().move_joints(joints_star, duration=coef*joints_movement, wait=wait)
        return SE3_to_pose(pose)

    
    def goto_poses(self, poses, dry_run, coef=3):
        joints = super().get_joints()
        traj = self.myIK.plan_trajectory(poses, joints)
        self.myIK.show_traj(traj, loop=dry_run)
        if not dry_run:
            for joints_star in traj:
                joints = super().get_joints()
                joints_movement = np.max(np.abs(joints - joints_star))
                print(f"joints movement {joints_movement}")
                super().move_joints(joints_star, duration=coef*joints_movement, wait=True)


def init_robot(ns='robot1', pose_path='/home/rmqlife/work/ur_slam/pose.json'):
    '''
    ns : robot topic name space
    '''
    transforms = [SE3.Rx(135, unit='deg'), SE3.Rx(45,unit="deg")]
    with open(pose_path, 'r') as file:
        data = json.load(file)
        arm_poses = [data['arm1_pose'], data['arm2_pose']]
        transforms = [pose_to_SE3(arm_poses[0]), pose_to_SE3(arm_poses[1])]
    # from hand_eye_calib import load_object
    if ns == "robot1":
        myIK = MyIK_rotate(transforms[0])
    elif ns == "robot2":
        myIK = MyIK_rotate(transforms[1])
    else:
        raise KeyError
    # base_transform = load_object("slam_data/images-20240731-100429/base_transform.pkl")
    return MyRobot_with_IK(myIK=myIK,ns=ns)  


if __name__ == "__main__":
    rospy.init_node('ik_step', anonymous=True)
    image_saver = MyImageSaver()
    framedelay = 1000//20

    robot = init_robot(ns='robot1')
    print(robot.get_joints())
    
    gripper = MyGripper()


    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        cam_pose = get_cam_pose(frame, intrinsics)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('k'):
            gripper.set_gripper(position=1000, force=2)
        if key == ord('j'):
            gripper.set_gripper(position=200, force=2)

        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            action = lookup_action(code)
            pose = robot.step(action=action, wait=False)
            print('robot pose', np.round(pose[:3], 3))
            if cam_pose is not None:
                print("cam pose", np.round(cam_pose[:3], 3))
            image_saver.record()

        
