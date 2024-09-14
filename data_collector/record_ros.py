import rospy
from sensor_msgs.msg import Image
from control_msgs.msg import JointTrajectoryControllerState
from message_filters import ApproximateTimeSynchronizer, Subscriber
import signal
import sys
import time
import h5py
import cv2
import numpy as np
import os
from cv_bridge import CvBridge

class info_saver:
    def __init__(self, frequency,robot_name):
        try:
            rospy.init_node('sync_listener', anonymous=True)
            print("Node initialized")
        except:
            print("ROS node has been initialized")

        self.data_buffer = {
            'header_seq': [], 'header_stamp': [],
            'joint_names': [],
            'desired_positions': [], 'desired_velocities': [],
            'actual_positions': [], 'actual_velocities': [],
            'error_positions': [], 'error_velocities': [],
            'image_data_seq1': [], 'depth_data_seq1': [],
            'image_data_seq2': [], 'depth_data_seq2': []
        }
        self.image_buffer = {
            'image_data1': [], 'depth_data1': [],
            'image_header_seq1': [], 'depth_header_seq1': [],
            'image_data2': [], 'depth_data2': [],
            'image_header_seq2': [], 'depth_header_seq2': []
        }
        self.frequency = frequency
        self.data_collected = False
        self.last_message_time = rospy.Time.now()
        self.robot_name = robot_name
    def info_save(self, image_msg1, image_msg2, joint_state_msg, depth_msg1, depth_msg2):
        self.last_message_time = rospy.Time.now()
        print("Received synchronized messages")
        # Extract timestamp
        timestamp = joint_state_msg.header.stamp
        print(timestamp.to_sec())
        # Extract position and velocity
        self.data_buffer['desired_positions'].append(joint_state_msg.desired.positions)
        self.data_buffer['desired_velocities'].append(joint_state_msg.desired.velocities)
        self.data_buffer['actual_positions'].append(joint_state_msg.actual.positions)
        self.data_buffer['actual_velocities'].append(joint_state_msg.actual.velocities)
        self.data_buffer['error_positions'].append(joint_state_msg.error.positions)
        self.data_buffer['error_velocities'].append(joint_state_msg.error.velocities)
        # Append header information
        self.data_buffer['header_seq'].append(joint_state_msg.header.seq)
        self.data_buffer['header_stamp'].append(timestamp.to_sec())
        self.data_buffer['joint_names'].append(joint_state_msg.joint_names)
        # Add data for camera1
        cv_image1 = self.imgmsg_to_cv2(image_msg1)  # Convert to OpenCV image
        cv_depth1 = self.imgmsg_to_cv2(depth_msg1)  # Convert to OpenCV image
        self.image_buffer['image_data1'].append(cv_image1)  # Store as NumPy array
        self.image_buffer['depth_data1'].append(cv_depth1)
        self.image_buffer['image_header_seq1'].append(image_msg1.header.seq)
        self.image_buffer['depth_header_seq1'].append(depth_msg1.header.seq)
        # Add data for camera2
        cv_image2 = self.imgmsg_to_cv2(image_msg2)  # Convert to OpenCV image
        cv_depth2 = self.imgmsg_to_cv2(depth_msg2)  # Convert to OpenCV image
        self.image_buffer['image_data2'].append(cv_image2)  # Store as NumPy array
        self.image_buffer['depth_data2'].append(cv_depth2)
        self.image_buffer['image_header_seq2'].append(image_msg2.header.seq)
        self.image_buffer['depth_header_seq2'].append(depth_msg2.header.seq)
        self.data_collected = True
        print(f"Data collected: {len(self.data_buffer['header_seq'])} samples")

        rospy.sleep(1/self.frequency)
    

    def save_image_jpg(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        if self.image_buffer['image_data1']:  # Check if there are images in the buffer
            image_data_path = file_path + '/image_data.jpg'
            cv2.imwrite(image_data_path, self.image_buffer['image_data1'][-1])  # Save the last image
            print(f"Image data saved in jpg: {image_data_path}")
        else:
            print("No image data to save.")


    def save_robot_state(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        state_data_path = file_path + '/robot_state.hdf5'
        with h5py.File(state_data_path, 'w') as f:
            for key, value in self.data_buffer.items():
                f.create_dataset(key, data=value)
        print(f"Data saved in hdf5: {state_data_path}")

    def save_image_buffer(self, file_path):
        os.makedirs(file_path, exist_ok=True)
        image_data_path = file_path + '/image_data.hdf5'
        with h5py.File(image_data_path, 'w') as f:
            for key, value in self.image_buffer.items():
                f.create_dataset(key, data=value)  # Store image data in a separate HDF5 file
        print(f"Image data saved in hdf5: {image_data_path}")

    def imgmsg_to_cv2(self, img_msg):
        # Convert ROS Image message to OpenCV format
        from cv_bridge import CvBridge
        bridge = CvBridge()
        output = bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        return np.array(output, dtype=np.float32)

    def listener(self):
        # camera1
        image_sub1 = Subscriber('/camera1/color/image_raw', Image)
        depth_sub1 = Subscriber('/camera1/depth/image_rect_raw', Image)
        # camera2
        image_sub2 = Subscriber('/camera2/color/image_raw', Image)
        depth_sub2 = Subscriber('/camera2/depth/image_rect_raw', Image)

        self.image_sub1 = image_sub1
        self.depth_sub1 = depth_sub1
        self.image_sub2 = image_sub2
        self.depth_sub2 = depth_sub2

        # Replace the topic existence checks with this function
        def topic_exists(topic_name):
            return any(topic_name in topic for topic, _ in rospy.get_published_topics())

        # Check if the topics exist
        robot_topic = '/' + self.robot_name + '/scaled_pos_joint_traj_controller/state'
        if not topic_exists(robot_topic):
            raise KeyError(f"{robot_topic} Topic does not exist")  
        else:
            print(f"{robot_topic} Topic exists")
        
        if not topic_exists('/camera1/color/image_raw'):
            raise KeyError("/camera1/...  Topic does not exist")
        else:
            print("/camera1 Topic exists")
        
        if not topic_exists('/camera2/color/image_raw'):
            raise KeyError("/camera2 Topic does not exist")
        else:
            print("/camera2 Topic exists")

        Subscriber_name ='/'+ self.robot_name + '/scaled_pos_joint_traj_controller/state'
        joint_state_sub = Subscriber(Subscriber_name, JointTrajectoryControllerState)
        self.joint_state_sub = joint_state_sub

        rospy.loginfo("Subscribers created")
        # 使用ApproximateTimeSynchronizer同步话题
        ats = ApproximateTimeSynchronizer([image_sub1, image_sub2, joint_state_sub, depth_sub1, depth_sub2], queue_size=10, slop=0.1)
        ats.registerCallback(self.info_save)
        
        print("Callback registered")
    
    def stop_subscriptions(self):
        # Unsubscribe from all topics
        self.image_sub1.unregister()
        self.depth_sub1.unregister()
        self.image_sub2.unregister()
        self.depth_sub2.unregister()
        self.joint_state_sub.unregister()
        print("Unsubscribed from all topics")
        


def signal_handler(sig, frame):
    print("Ctrl+C pressed. Saving data...")
    # Save the data
    time_str = time.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join('data', time_str)
    if info_saver.data_collected:
        info_saver.save_robot_state(file_path)
        info_saver.save_image_buffer(file_path)
        # info_saver.save_image_jpg(file_path)
        print("Data saved. Exiting...")
    else:
        print("No data collected. Exiting without saving.")
    sys.exit(0)




if __name__ == "__main__":
    info_saver = info_saver(10,"robot1")
    info_saver.listener()
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()