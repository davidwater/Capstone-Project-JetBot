import numpy as np
import cv2
import yaml
# Parameters
# TODO : Read from file
n_row = 6
n_col = 9
n_min_img = 10  # img needed for calibration
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria
corner_accuracy = (11, 11)
result_file = "./calibration.yaml"

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(n_row-1,n_col-1,0)
objp = np.zeros((n_row * n_col, 3), np.float32)
objp[:, :2] = np.mgrid[0:n_row, 0:n_col].T.reshape(-1, 2)


def gstreamer_pipeline(
    w=640,
    h=480,
    display_width=640,
    display_height=480,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            w,
            h,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    camera = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if camera.isOpened():
        window_handle = cv2.namedWindow("calibration", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("calibration", 0) >= 0:
            ret, img= camera.read()
            cv2.imshow("calibration", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            def usage():
                print("Press on displayed window:")
                print("[space]     : take picture")
                print("[c]         : compute calibration")
                print("[r]         : reset program")
                print("[ESC]    : quit")

            usage()           
            Initialization = True

            while True:
                if Initialization:
                    print("Initialize data structures ..")
                    objpoints = []  # 3d point in real world space
                    imgpoints = []  # 2d points in image plane.
                    n_img = 0
                    Initialization = False
                    tot_error = 0

                # Read from camera and display on windows
                ret, img= camera.read()
                cv2.imshow("calibration", img)

                if not ret:
                   print("Cannot read camera frame, exit from program!")
                   camera.release()
                   cv2.destroyAllWindows()
                   break
                # Wait for instruction
                k = cv2.waitKey(50)

                # SPACE pressed to take picture
                if k % 256 == 32:
                    print("Adding image for calibration...")
                    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Find the chess board corners
                    ret, corners = cv2.findChessboardCorners(imgGray, (n_row, n_col), None)

                    # If found, add object points, image points (after refining them)
                    if not ret:
                       print("Cannot found Chessboard corners!")

                    else:
                       print("Chessboard corners successfully found.")
                       objpoints.append(objp)
                       n_img += 1
                       corners2 = cv2.cornerSubPix(imgGray, corners, corner_accuracy, (-1, -1), criteria)
                       imgpoints.append(corners2)

                    # Draw and display the corners
                       imgAugmnt = cv2.drawChessboardCorners(img, (n_row, n_col), corners2, ret)
                       cv2.imshow('Calibration', imgAugmnt)
                       cv2.waitKey(500)

                    # "c" pressed to compute calibration
                elif k % 256 == 99:
                   if n_img <= n_min_img:
                       print("Only ", n_img, " captured, ", " at least ", n_min_img, " images are needed")

                   else:
                       print("Computing calibration ...")
                       ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640, 480), None, None)
                       if not ret:
                           print("Cannot compute calibration!")

                       else:
                           print("Camera calibration successfully computed")
                           # Compute reprojection errors
                           for i in range(len(objpoints)):
                               imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                               error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                               tot_error += error
                           print("Camera matrix: ", mtx)
                           print("Distortion coeffs: ", dist)
                           print("Total error: ", tot_error)
                           print("Mean error: ", np.mean(error))

                           # Saving calibration matrix
                           print("Saving camera matrix .. in ", result_file)
                           data = {"camera_matrix": mtx.tolist(), "dist_coeff": dist.tolist()}
                           with open(result_file, "w") as f:
                              yaml.dump(data, f, default_flow_style=False)
                # ESC pressed to quit
                elif k % 256 == 27:
                   print("Escape hit, closing...")
                   camera.release()
                   cv2.destroyAllWindows()
                   break
                # "r" pressed to reset
                elif k % 256 == 114:
                   print("Reset program...")
                   Initialization = True

                



            # Stop the program on the ESC key
            if keyCode == 27:
                break
        camera.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
if __name__ == "__main__":
    show_camera()












