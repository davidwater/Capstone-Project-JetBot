import cv2
import numpy as np
from jetbot import Robot
robot= Robot()
from numpy import size


def do_Red_mask():
    red_lower_bound = np.array([160, 80, 150])
    red_upper_bound = np.array([250, 255, 255])
    Rmask = cv2.inRange(hsv, red_lower_bound, red_upper_bound)
    red = cv2.bitwise_and(frame, frame, mask=Rmask)
    red_circles = cv2.HoughCircles(Rmask, cv2.HOUGH_GRADIENT, 1, 80,
                                   param1=50, param2=10, minRadius=8, maxRadius=14)
    if red_circles is not None:
        red_circles = np.uint16(np.around(red_circles))

        for i in red_circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(Rmask, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(frame,'Red',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
    return red_circles

def do_Green_mask():
    Green_lower_bound = np.array([55, 30, 30])
    Green_upper_bound = np.array([110, 255, 255])
    Gmask = cv2.inRange(hsv, Green_lower_bound, Green_upper_bound)
    green = cv2.bitwise_and(frame, frame, mask=Gmask)
    green_circles = cv2.HoughCircles(Gmask, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=8, maxRadius=14)
    if green_circles is not None:
        green_circles = np.uint16(np.around(green_circles))

        for i in green_circles[0, :]:

                cv2.circle(frame, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(Gmask, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(frame,'Green',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
    return green_circles

def do_yellow_mask():
    Yellow_lower_bound = np.array([7, 0, 0])
    Yellow_upper_bound = np.array([20, 255, 255])
    Ymask = cv2.inRange(hsv, Yellow_lower_bound, Yellow_upper_bound)
    yellow = cv2.bitwise_and(frame, frame, mask=Ymask)
    yellow_circles = cv2.HoughCircles(Ymask, cv2.HOUGH_GRADIENT, 1, 30,
                                      param1=50, param2=10, minRadius=8, maxRadius=14)
    if yellow_circles is not None:
        yellow_circles = np.uint16(np.around(yellow_circles))

        for i in yellow_circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(Ymask, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(frame,'Yellow',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
    return yellow_circles

cap = cv2.VideoCapture(0)


while cap.isOpened(0):


    ret, frame = cap.read()
    if not ret:
        print("No detected frame")
        break
    font = cv2.FONT_ITALIC
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    Redcircle = do_Red_mask()
    Greencircle = do_Green_mask()
    Yellowcircle = do_yellow_mask()

    if Greencircle == True:
        robot.set_motor(0.1,0.1)
    else:
         robot.stop()


    cv2.imshow('Result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()