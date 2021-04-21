'''
Rersource:
Detection of ArUco Markers : https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
OpenCV Camera Calibration : https://medium.com/@aliyasineser/opencv-camera-calibration-e9a48bdd1844
ArUco Marker Tracking with : OpenCV: https://medium.com/@aliyasineser/aruco-marker-tracking-with-opencv-8cb844c26628
OpenCV相机标定与畸变校正 : https://www.sohu.com/a/343521135_823210
'''
'''
Edit: 
1. the resource above are using image, but I change to the straming camera for the phone camera capture.
2. Sample size (number of image) is to large using stream, so I add a count to limit the sample size to 30.
'''

import numpy as np
import cv2
import glob
URL = "http://192.168.1.77:8080/video"

video_capture = cv2.VideoCapture(URL)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 10)
video_capture.set(cv2.CAP_PROP_FPS, 10)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
def calibrate(square_size = 0.015, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    gray = None
    count=0
    while(True):
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit()
        resizedFrame = cv2.resize(frame, (360, 640))
        gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            resizedFrame = cv2.drawChessboardCorners(resizedFrame, (width, height), corners2, ret)
        if cv2.waitKey(1) & 0xFF == ord('q') or count==30:
            break
        count+=1
        cv2.imshow('frame', resizedFrame)
    video_capture.release()
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return camera_matrix, dist_matrix
'''
if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate()
    save_coefficients(mtx, dist, "./camera.yml")
    print("Calibration is finished. RMS: ", ret)
'''