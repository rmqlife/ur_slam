import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        print('ready')
        self.save_rgb = False
        self.save_depth = False
        self.count = 0
        self.folder_path = "images"+time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def rgb_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            if self.save_rgb:
                self.save_image(cv_image, "rgb")
                rospy.loginfo("RGB image saved successfully!")
                self.save_rgb = False
        except Exception as e:
            rospy.logerr("Error saving RGB image: %s", str(e))

    def depth_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            if self.save_depth:
                self.save_image(cv_image, "depth")
                rospy.loginfo("Depth image saved successfully!")
                self.save_depth = False
        except Exception as e:
            rospy.logerr("Error saving depth image: %s", str(e))

    def generate_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_image(self, image, prefix):
        image_filename = os.path.join(self.folder_path, f"{prefix}_{self.count}.png")
        cv2.imwrite(image_filename, image)

    def record(self, rgb=True, depth=True):
        self.save_rgb = rgb
        self.save_depth = depth
        self.count += 1

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node('image_saver')
        import time
        image_saver = ImageSaver()
        # Example usage: Save RGB and depth images
        while not rospy.is_shutdown():
            image_saver.record()  # Save images
            time.sleep(1)  # Sleep for 1 seconds
        image_saver.spin()

    except rospy.ROSInterruptException:
        pass
