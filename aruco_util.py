import cv2
import cv2.aruco as aruco
import numpy as np
# MAKE SURE pip install opencv-contrib-python==4.6.0.66
def detect_aruco(image, verbose=False):
    # Load the image    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) #DICT_6X6_250
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
    # Draw detected markers on the image
    image_with_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)
    
    # Display the image with detected markers
    if verbose:
        cv2.imshow('Detected ArUco Markers', image_with_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # take first corners
    # print(corners[0][0])
    print(f'detected {corners[0].shape[0]} aruco(s)', )
    corner = corners[0][0]
    center = np.floor(np.mean(corner, axis=1)).astype('int')
    return corner

if __name__=="__main__":
    # Path to the image containing ArUco markers
    image_path = '/media/rmqlife/DM/ur_slam/0612-facedown/rgb_14.png'
    image = cv2.imread(image_path)
    # Call the function to detect ArUco markers
    corner = detect_aruco(image, True)
    print(corner)