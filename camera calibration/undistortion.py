import numpy as np
import cv2
import yaml

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
        window_handle = cv2.namedWindow("undistortion", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("undistortion", 0) >= 0:
            ret, img= camera.read()
            cv2.imshow("undistortion", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF

            while(True):

                  ret, img = camera.read()

                  if not ret:
                      print("Can't receive frame (stream end?). Exiting ...")
                      break
                  with open('calibration.yaml', 'r') as f:
                            data = yaml.load(f, Loader=yaml.CLoader)
                  cameraMatrix = data['camera_matrix']
                  dist = data['dist_coeff']
                  camera_matrix = np.array(cameraMatrix)
                  dist_coeff = np.array(dist)

                  newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (640, 480), 1, (640, 480), )

                  dst = cv2.undistort(img, camera_matrix, dist_coeff, None, newCameraMatrix)
                  x, y, w, h = roi
                  dst = dst[y:y + h, x:x + w]
                  
                  
                  cv2.imshow('undistortion', dst)

                  if cv2.waitKey(1) == ord('q'):
                     break


        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()



