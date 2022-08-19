import cv2
import numpy as np
import cv2.aruco as aruco
from math import sqrt, pow, atan, acos, pi, atan2
from numpy.linalg import inv, norm
from jetbot import Robot
import time
# file = np.load('calibration.npz')

class detection:
    def __init__(self, original_frame, file):
        # original camera frame without distortion
        self.original_frame = original_frame
        # undistorted frame
        self.undis_frame = None
        # grayscale frame
        self.gray = None
        # camera parameter
        mtx, dist, R, T = [file[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
        self.mtx = mtx
        self.dist = dist
        self.new_mtx = None
        # perspective transform
        self.warp_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None
        self.roi_lane = np.array([[0, 480], [640, 480], [640, 250], [0, 250]])
        self.dst_pts = np.float32([[0, 400], [400, 400], [400, 0], [0, 0]])
                                
    
    def gstreamer_pipeline(
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
    ):
        return (
            "nvarguscamerasrc sensor-id=%d !"
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
            )
        )
    

    def undistortion(self, frame):
        mtx = self.mtx
        dist = self.dist
        h1 = self.original_frame.shape[0]
        w1 = self.original_frame.shape[1]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
        self.new_mtx = newcameramatrix
        undist_image = cv2.undistort(frame, mtx, dist, None, newcameramatrix)
        x, y, w1, h1 = roi
        dst1 = undist_image[y:y + h1, x:x + w1]
        undist_image = cv2.resize(dst1, (640, 480), cv2.INTER_NEAREST)
        self.undis_frame = dst1
        return undist_image

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.gray = gray
        return gray

    def aruco_detect(self, gray):
        mtx = self.mtx
        dist = self.dist
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        back = None
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.03, mtx, dist)
            for i in range(len(ids)):
                x = tvec[i, 0, 0]
                y = tvec[i, 0, 1]
                z = tvec[i, 0, 2]
                R = cv2.Rodrigues(rvec[i])[0]
                # euler angle
#                 tx = atan2(R[2, 1], R[2, 2])
#                 ty = atan2(-R[2, 0], sqrt(pow(R[2, 1], 2) + pow(R[2, 2], 2)))
                tz = np.rad2deg(atan2(R[1, 0], R[0, 0]))
#                 angle = np.rad2deg(np.array([tx, ty, tz]))
#                 print(tz)
                dis = sqrt(pow(x,2)+pow(y,2)+pow(z,2))*100 + 3
                # modified distance
                dis = 0.0084*pow(dis, 2) + 0.5877*dis - 2.1246
#                 print('id = ', ids[i], dis, '(cm)')
                id = int(ids[i])
                if (0 <= id < 22) & (tz < 90):
                    back = np.array([id, dis])
#                 distance[i, :] = np.array([id, dis, tz])
#         distance = np.round(distance, 1)
        return back

    def stop_line_detect(self, threshold):
        frame = self.original_frame
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower = np.array([51, 107, 139])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        mask = mask[300:480, :]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (320, 4))
        mask = cv2.dilate(mask, None, iterations=2)
#         cv2.imshow('mask', mask)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        num = np.transpose(np.nonzero(mask))
        if len(num) > threshold:
            detect = True
        else:
            detect = False
        return detect
    

    def HOG_detect(self):
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        people = False
        distortion = self.original_frame
        gray = cv2.cvtColor(distortion, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240), cv2.INTER_NEAREST)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        if len(boxes) != 0:
            people = True
        return people

    def traffic_light_detect(self):
        frame = self.undis_frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Green_lower_bound = np.array([54, 20, 20])
        Green_upper_bound = np.array([85, 255, 255])
        Gmask = cv2.inRange(hsv, Green_lower_bound, Green_upper_bound)
        green_circles = cv2.HoughCircles(Gmask, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=8, maxRadius=14)
        if green_circles is not None:
            green = True
        else:
            green = False
        return green
    
    def avoid_people(self, robot):
        robot.set_motors(-0.1,0.15)
        time.sleep(0.65)
        robot.set_motors(0.2,0.12)
        time.sleep(1.5)
        robot.set_motors(-0.1,0.15)
        time.sleep(0.65)
        robot.set_motors(0,0)
        
    def turn_left(self, robot):
        robot.set_motors(0.11,0.185)
        time.sleep(4.2)
        robot.set_motors(0.101,0.108)
        time.sleep(2)
        robot.stop()

        
    def turn_right(self, robot):
        robot.set_motors(0.154,0.15)
        time.sleep(0.15)
        robot.set_motors(0.215, 0.1)
        time.sleep(2.225)
        robot.set_motors(0.101,0.108)
        time.sleep(2)
        robot.stop()
        
    def go_straight(self, robot):
        robot.set_motors(0.101,0.108)
        time.sleep(3)

