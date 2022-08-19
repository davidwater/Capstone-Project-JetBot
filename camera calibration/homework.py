import numpy as np
import cv2 as cv
import glob


# Load previously saved data
##################################TO DO##################################

#IN Calibration.py, use numpy function to get ur npz file

with np.load('calibration.npz') as file:


##################################TO DO##################################

    mtx, dist, rvecs, tvecs = [file[i] for i in ('mtx','dist','rvecs','tvecs')]


def draw(img, corners, imgpts):
    corners = corners.astype(int)
    imgpts = imgpts.astype(int)
    corner = tuple(corners[34].ravel())
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (0,0,255), 5)

    return img



criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
objp = objp[::-1,:]
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
n_img = 1

#for image in glob.glob('calibrationimage\*.jpg'):
for f_name in glob.glob('calibration_{}.jpg'.format(n_img)):
    img = cv.imread(f_name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (7,5),None)

    if ret == True:

        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

        # Find the rotation and translation vectors.
        _ , r_vects, t_vects = cv.solvePnP(objp, corners2, mtx, dist)




        # Project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, r_vects, t_vects, mtx, dist)


        img = draw(img,corners2,imgpts)
        cv.imshow('ur_file_name',img)

        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            cv.imwrite('ur_image.jpg', img)

r_vects, _ = cv.Rodrigues(r_vects)
cv.destroyAllWindows()