import cv2
import numpy as np

# Define the size of the calibration pattern (e.g., a checkerboard)
pattern_size = (6, 8)  # Number of inner corners (points) in the checkerboard

# Criteria for corner detection
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Initialize video capture from the camera
cap = cv2.VideoCapture(0)  # Replace 0 with the camera index if using multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    chessboard_ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # If corners are found, add object points and image points
    if ret:
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        objpoints.append(objp)
        corners_subpixel = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_subpixel)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, pattern_size, corners_subpixel, ret)
        cv2.imshow('Calibration', frame)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print intrinsic parameters (camera matrix)
print("Camera Matrix:")
print(mtx)

# Print distortion coefficients
print("\nDistortion Coefficients:")
print(dist)

# # Optionally, undistort an example image
# example_image = cv2.imread('calibration_image.jpg')
# undistorted_image = cv2.undistort(example_image, mtx, dist)

# # Display the undistorted image (optional)
# cv2.imshow('Undistorted Image', undistorted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
