'''
Rersource:
Detection of ArUco Markers: https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
OpenCV Camera Calibration: https://medium.com/@aliyasineser/opencv-camera-calibration-e9a48bdd1844
ArUco Marker Tracking with OpenCV: https://medium.com/@aliyasineser/aruco-marker-tracking-with-opencv-8cb844c26628
estimatePoseSingleMarkers() function reform: https://home.gamer.com.tw/creationDetail.php?sn=4093935
python cv2.Rodrigues() function: https://blog.csdn.net/qq_40475529/article/details/89409303
'''

import cv2
import numpy as np
from camera_calibration import calibrate, save_coefficients, load_coefficients
URL = "http://192.168.1.77:8080/video"

class marker_camera():
    def __int__(self):
        self.video_capture = None

    def video_capture(self):
        self.video_capture =cv2.VideoCapture(URL)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        self.video_capture.set(cv2.CAP_PROP_FPS, 10)

    def read_marker(self):
        camera_matrix, dist_matrix = load_coefficients("./camera.yml")
        while(self.video_capture is not None):
            ret, frame = self.video_capture.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                exit()
            resizedFrame = cv2.resize(frame, (360, 640))
            gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
            # Detect the markers in the image
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(gray, cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50))
            if(markerIds is not None):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.03, camera_matrix, dist_matrix)
                cv2.aruco.drawAxis(resizedFrame, camera_matrix, dist_matrix, rvec, tvec, 0.03)
            else:
                print("nothing")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('frame', resizedFrame)
        # When everything done, release the capture
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera = marker_camera()
    camera.video_capture()
    camera.read_marker()




