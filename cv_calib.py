import cv2 
import numpy as np 
  
hand_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [ 
                       1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) 
  
eye_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 
                       [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]) 
  
# rotation matrix between the target and camera 
R_target2cam = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [ 
                        0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) 
  
# translation vector between the target and camera 
t_target2cam = np.array([0.0, 0.0, 0.0, 0.0]) 
  
# transformation matrix 
T, _ = cv2.calibrateHandEye(hand_coords, eye_coords)

print(T)