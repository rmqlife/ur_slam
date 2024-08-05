PGI 140 80 gripper

1. To connect the control box of UR5-CB3, make sure the digital output 1 is enabled in the TouchPad

2. connect USB to computer's port, and use https://github.com/DH-Robotics/dh_gripper_ros

3. make sure the remove AG95 to compile with catkin_make

launch file is :

roslaunch dh_gripper_driver dh_gripper.launch

to publish:

rostopic pub /gripper/ctrl dh_gripper_msgs/GripperCtrl "initialize: false
position: 800.0
force: 20.0
speed: 50.0"
