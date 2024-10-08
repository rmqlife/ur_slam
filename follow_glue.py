from utils.lightglue_util import MyGlue
from ik_step import *
from ros_utils import 
if __name__=="__main__":
    rospy.init_node('follow_aruco')
    image_saver = MyImageSaver()
    rospy.sleep(1)
    framedelay = 1000//20

    robot = init_robot(ns='robot1')

    glue = MyGlue(match_type="LightGlue") # Aruco LightGlue
    goal_frame = None

    src_pts = None



    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        show_frame = frame.copy() # in case of frame is ruined by painting frame
        depth = image_saver.depth_image

        if goal_frame is None:
            goal_frame = frame.copy()
            goal_depth = depth.copy()

        src_pts, dst_pts = glue.match(frame, goal_frame)
        src_pts = [pt.tolist() for pt in src_pts] 
        dst_pts = [pt.tolist() for pt in dst_pts] 
        for (x1, y1), (x2, y2) in zip(src_pts, dst_pts):
            cv2.line(show_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)

        cv2.imshow('Camera', show_frame)
        # Exit on 'q' key press
        key = cv2.waitKey(framedelay) & 0xFF 
        if key == ord('q'):
            break

        elif key in key_map:
            code  = key_map[key]
            print(f"action {code}")
            action = lookup_action(code)
            pose = robot.step(action=action, wait=False)
            print('robot pose', np.round(pose[:3], 3))

        if key == ord('m'):      

            src_pts_3d, dst_pts_3d = glue.map_3d_pts(src_pts, dst_pts, depth, goal_depth, intrinsics)
            print("src_pts_3d", src_pts_3d.shape)
            print(src_pts_3d)
            
            print("dst_pts_3d")
            print(dst_pts_3d)
            # for lightglue
            R, t = compute_rigid_transform(src_pts_3d, dst_pts_3d, enable_R=False)
            move = t

            # move[2]=0
            move = SE3([move[1], move[0], move[2]])
            move.printline()
            
            robot.step(action=move, wait=False)

        if key == ord('g'):
            # setup goal
            goal_frame = frame.copy()
            goal_depth = depth.copy()
            print('set goal')
            
    cv2.destroyAllWindows()