import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os


class MyImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.count = 0
        self.folder_path = "data/images"+time.strftime("-%Y%m%d-%H%M%S")
        rospy.sleep(1)
        print(f'init MyImageSaver at {self.folder_path}')

    def rgb_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Error saving RGB image: %s", str(e))

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Error saving depth image: %s", str(e))

    def generate_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_image(self, image, prefix):
        os.makedirs(self.folder_path, exist_ok=True)
        image_filename = os.path.join(self.folder_path, f"{prefix}_{self.count}.png")
        cv2.imwrite(image_filename, image)
        print(f"write to {image_filename}")

    def record(self):
        self.save_image(self.rgb_image, "rgb")
        self.save_image(self.depth_image, 'depth')
        self.count += 1

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        rospy.init_node('image_saver')
        import time
        image_saver = MyImageSaver()
        time.sleep(1)
        # Example usage: Save RGB and depth images
        while not rospy.is_shutdown():
            image_saver.record()  # Save images
            time.sleep(1)  # Sleep for 1 seconds
        image_saver.spin()

    except rospy.ROSInterruptException:
        pass
