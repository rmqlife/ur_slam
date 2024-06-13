import cv2.aruco as aruco
import numpy as np
import cv2

Dist = np.array([0.14574457705020905, -0.49072766304016113, 0.0002240741887362674, -0.00014576673856936395, 0.4482966661453247])  # system given

mtx=np.array([[901.964599609375, 0.0, 652.5621337890625],
 [  0.       ,  902.0592651367188, 366.7032165527344],
 [  0.,           0.,           1.        ]])

class ArucoDetector:
    def __init__(self):
        self.parameters = aruco.DetectorParameters_create()
        self.ids = None
        self.rvec = None
        self.tvec = None

    def detect(self, image, aruco_dict, aruco_size=0.02):
        corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=self.parameters)
        if ids is not None:
            self.ids = ids
            self.rvec, self.tvec, _ = aruco.estimatePoseSingleMarkers(corners, aruco_size, mtx, Dist)
            return True
        else:
            return False

    def draw(self, image):
        for i in range(self.rvec.shape[0]):
            cv2.drawFrameAxes(image, mtx, Dist, self.rvec[i, :, :], self.tvec[i, :, :], 0.03)
            # aruco.drawDetectedMarkers(image, corners)
            cv2.putText(image, "Id: " + str(self.ids), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Example usage:
if __name__ == "__main__":
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    detector = ArucoDetector()
    image = cv2.imread('/media/rmqlife/DM/ur_slam/0612-facedown/rgb_31.png')  # Provide the path to your image
    print(image.shape)
    detected = detector.detect(image, aruco_dict)

    # detector.draw(image)
    cv2.imshow("Aruco Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("No ArUco marker detected.")
