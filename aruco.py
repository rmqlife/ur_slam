#!/usr/bin/env python
from typing import Any
import numpy as np
import cv2
import pandas as pd
import cv2.aruco as aruco
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

Dist = np.array([0.14574457705020905, -0.49072766304016113, 0.0002240741887362674, -0.00014576673856936395, 0.4482966661453247])  # System given

mtx = np.array([[901.964599609375, 0.0, 652.5621337890625],
                [0.0, 902.0592651367188, 366.7032165527344],
                [0.0, 0.0, 1.0]])

class RosCamera:
    def __init__(self):
        self.real_camera_img_aligned = None
        self.bridge = CvBridge()

    def camera_image_callback(self, msg):
        self.real_camera_img_aligned = msg

    def get_image(self):
        image = self.real_camera_img_aligned
        if image is not None:
            cv_img = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            return cv_img
        else:
            return None

    def display(self, collect=False):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        aruco_dict2 = aruco.Dictionary_get(aruco.DICT_6X6_250)
        filename = "endpose_data.csv"
        img_path = './charuco_test/'
        epi = 0

        while True:
            image = self.get_image()
            if collect is False:
                # Perform Aruco pose estimation if not collecting
                aruco.PoseEstimate(image, aruco_dict, draw=True)
                aruco.PoseEstimate(image, aruco_dict2, aruco_size=0.022, draw=True)

            cv2.imshow('RealSense', image)
            key = cv2.waitKey(1)

            if key == ord('q'):
                break

            if (key == ord('s') or key == ord('S')):
                epi += 1
                imgname = str(epi) + ".jpg"
                cv2.imwrite(img_path + imgname, image)

                if epi != 1:
                    df_old = pd.read_csv(filename)
                    df = pd.DataFrame([end_position + end_orientation], columns=['position x', 'position y', 'position z', 'Rotdata x', 'Rotdata y', 'Rotdata z', 'Rotdata w'])
                    df = pd.concat([df_old, df], ignore_index=True)
                else:
                    df = pd.DataFrame([end_position + end_orientation], columns=['position x', 'position y', 'position z', 'Rotdata x', 'Rotdata y', 'Rotdata z', 'Rotdata w'])

                df.to_csv(img_path + filename, index=False)

if __name__ == "__main__":
    rl_camera = RosCamera()
    print("If it's used for calibration, please enter 1, or 0")
    user_input = input("Enter 1 or 0:")

    if user_input == "1":
        print("Press 's' to save image")
        collect_images = True
    elif user_input == "0":
        print("Calibration mode is disabled.")
        collect_images = False
    else:
        print("Please re-launch if you plan to calibrate")
        collect_images = False

    print("Press 'q' to exit")
    rl_camera.display(collect=collect_images)
