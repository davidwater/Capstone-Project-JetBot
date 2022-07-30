import numpy as np
import cv2 as cv
import math
from robot import Robot
mtx = np.array([[401.84357792238274,0,347.77201697589762],[0,537.2819150317955,253.54949838288],[0,0,1]])
rmtx = np.array([[-0.99876833,-0.03371205,-0.03640483],[-0.01583112,-0.47884724,0.87775549],[-0.04702329,0.87725072,0.47772376]])
tvecs = np.array([2.88221901,0.71595076,11.14039253])

robot = Robot()


def imgpt2wrd(imgcoor):
    img1x3 = np.array([imgcoor[0], imgcoor[1], 1])
    mtxInv = np.linalg.inv(mtx)

    # Rot matrix
    RInv = np.linalg.inv(rmtx)

    # RInv * mtxInv
    RMinv = np.dot(RInv,mtxInv)
    img3x1 = img1x3.reshape((3,1))

    # solve for s
    left = np.dot(RMinv, img3x1)
    right = np.dot(RInv, tvecs.T)
    s = right[2] / left[2]

    # cal 3D coor
    X = s * left - right
    Xtruth = 2.3 * X
    return Xtruth





def ctrl_mixer(r):
    #Vx=0.05
    Ainv=np.array([[0.4,2.0834],[0.4,-2.0834]])
    U=np.array([450,r])
    B=np.dot(Ainv,U.T)*3.14/180
    RPSR=B[0]
    RPSL=B[1]
    return RPSR,RPSL


def motor_ctrl(B1,B2):
    left_V=(B2+1.07)/50.3 
    right_V=(B1+1.26)/50.3 
    robot.set_motors(left_V, right_V)


def motor_ctrl_cur(B1,B2):
	left_V=(B2+1.07)/50.3
	right_V=(B1+1.26)/50.3
	robot.set_motors(left_V, right_V)
