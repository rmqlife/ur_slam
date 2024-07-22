import numpy as np
import roboticstoolbox as rtb
from utils.pose_util import *
from spatialmath import SE3, SO3
from myIK import MyIK
import rospy
from std_msgs.msg import Float32
import cv2
from follow_aruco import *


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


def init_real_robot():
    import sys
    sys.path.insert(0,'/home/rmqlife/work/catkin_ur5/src/teleop/src')
    from myRobot import MyRobot
    robot=MyRobot()
    return robot

def lookup_action(code, t_move=0.03, r_move=5):
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


def act_by_code(robot, action_code, wait):

    joints = robot.get_joints()
    myIK = MyIK(use_ikfast=False)
    pose = myIK.fk(joints)
    pose_se3 = pose_to_SE3(pose)
    tcp_move = lookup_action(action_code)

    # tcp_move = SE3.Rx(135, unit='deg') * tcp_move

    if abs(action_code)<=3:
        pose_se3 = tcp_move * pose_se3
    else:
        # keep the trans
        t =  pose_se3.t
        # tcp_move = SE3.Ry(45, unit='deg') * tcp_move
        pose_se3 = tcp_move * pose_se3
        pose_se3.t = t


    print("pose_se3", pose_se3)
    joints_star = myIK.ik_se3(pose_se3, q=joints)
    # compute the difference between joints and joints_star
    joints_movement = np.max(np.abs(joints - joints_star))
    print("joints movement {joints_movement}")
    robot.move_joints(joints_star, duration=5*joints_movement, wait=wait)
    return joints_star, SE3_to_pose(pose_se3)

if __name__ == "__main__":
    rospy.init_node('ik_step', anonymous=True)
    robot = init_real_robot()   
    image_saver = MyImageSaver()
    intrinsics = load_intrinsics("slam_data/intrinsics_d435.json")
    framedelay = 1000//20

    i = 0
    while not rospy.is_shutdown():

        frame = image_saver.rgb_image
        cam_pose = get_cam_pose(frame, intrinsics)
        print(cam_pose)
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break
        if key == ord('s'):
            image_saver.record()
        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            act_by_code(robot, action_code=code)

        # i += 1
        # # build action
        # # test actions
        # code = 4
        # act_by_code(robot, action_code=code)
        # print(f"action {code}")

        
