# coding=UTF-8
import cv2
import numpy as np
from numpy.linalg import inv, norm
from math import acos, pi, atan
import PID_v2_1 as Pv
import PID_v2_2 as PID
import yaml
with np.load('calibration.npz') as file:
    mtx, dist, rvecs, tvecs = [file[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
with open('calibration.yaml', 'r') as f:
	data = yaml.load(f, Loader=yaml.CLoader)
	cameraMatrix = data['camera_matrix']
	dist = data['dist_coeff']
	camera_matrix = np.array(cameraMatrix)
	dist_coeff = np.array(dist)



def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    ret3, binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary_img


def find_line(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 8
    # Set height of windows
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height  # window上下界
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) > 3000:
        if len(rightx) > 3000:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            loc_mid, end_mid = draw_area(left_fit, right_fit)
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            virtual_pt = np.array([[binary_warped.shape[1], 0], [binary_warped.shape[1], binary_warped.shape[0]]],
                                  dtype=np.int32)
            loc_mid, end_mid = draw_line(left_fit, virtual_pt)
    else:
        if len(rightx) > 3000:
            right_fit = np.polyfit(righty, rightx, 2)
            virtual_pt = np.array([[0, 0], [0, binary_warped.shape[0]]], dtype=np.int32)
            loc_mid, end_mid = draw_line(right_fit, virtual_pt)
        else:
            loc_mid = np.array([320, 479])
            end_mid = np.array([320, 0])

    return loc_mid, end_mid


def draw_line(fitcoefficient,virtual_pt):
    #end mid pt
    endy=100
    endx=fitcoefficient[0]*endy**2 + fitcoefficient[1]*endy + fitcoefficient[2]
    endpt=np.array([[endx,endy]])
    endpts=np.vstack((virtual_pt[0],endpt))
    end_mid=np.mean(endpts,0,dtype=np.int32)
    #loc mid pt
    locy=479
    locx=fitcoefficient[0]*locy**2 + fitcoefficient[1]*locy + fitcoefficient[2]
    locpt=np.array([[locx,locy]])
    #print(f'virtual pt:\n{virtual_pt[1]}\nlocpt:\n{locpt}')
    locpts=np.vstack((virtual_pt[1],locpt))
    loc_mid=np.mean(locpts,0,dtype=np.int32)
    return loc_mid,end_mid


def draw_area(left_fit, right_fit):
    #end mid pt
    endy=100
    left_xe = left_fit[0]*endy**2 + left_fit[1]*endy + left_fit[2]
    right_xe = right_fit[0]*endy**2 + right_fit[1]*endy + right_fit[2]
    leftpte=np.array([[left_xe,endy]])
    rightpte=np.array([[right_xe,endy]])
    endpts=np.vstack((leftpte,rightpte))
    end_mid=np.mean(endpts,0,dtype=np.int32)
    #loc mid
    locy=479
    left_xl = left_fit[0]*locy**2 + left_fit[1]*locy + left_fit[2]
    right_xl = right_fit[0]*locy**2 + right_fit[1]*locy + right_fit[2]
    leftptl=np.array([[left_xl,locy]])
    rightptl=np.array([[right_xl,locy]])
    locpts=np.vstack((leftptl,rightptl))
    loc_mid=np.mean(locpts,0,dtype=np.int32)
    return loc_mid,end_mid


def back_perspective(Minv, loc_mid, end_mid):
    # loc
    ul = loc_mid[0]
    vl = loc_mid[1]
    pl = Minv.dot([[ul], [vl], [1]])
    pl = (int(pl[0]), int(pl[1]))
    # end
    pe = Minv.dot([[end_mid[0]], [end_mid[1]], [1]])
    pe = (int(pe[0]), int(pe[1]))
    return pl, pe


def detect_angle(p_loc, p_end, img):
	#p_loc = im2b(p_loc[0], p_loc[1])
	#p_end = im2b(p_end[0], p_end[1])
    e1 = np.array([p_end[0]-p_loc[0], p_end[1] - p_loc[1]]).astype('float32')
    o = np.array([p_loc[0], p_loc[1]])
    A = (o[0], o[1])
    b = np.array([p_loc[0], 320])
    B = (b[0], b[1])
    #
    e2 = b-o
    e = np.dot(e1, e2)/(norm(e1)*norm(e2))
    angle = atan(e1[1]/e1[0])
    angle = angle*180/pi
    if abs(angle) < 3:
		angle = 0
    #
    cv2.line(img, p_end, p_loc, (0, 255, 255), 3)
    img = cv2.line(img, A, B, (0, 255, 0), 3)
    return img, angle


def b2im(x, y, z, mtx):
    w = np.array([[x-5.74], [y+2.83], [z], [1]])
    R = np.array([[0.04306311, -0.99895377, -0.01539257],
                  [-0.52110037, -0.00931324, -0.85344459],
                  [0.85240833, 0.04477305, -0.52095624]])
    T = np.array([[3.7615314],
                  [1.8150472],
                  [8.94055666]])
    RT = np.hstack([R, T])
    im = mtx.dot(RT.dot(w))
    u_b = int(im[0] / im[2])
    v_b = int(im[1] / im[2])
    s = im[2]
    return u_b, v_b, s
    
    
def im2b(u, v):
	im = np.array([[u], [v], [1]])
	R = np.array([[0.03757486, -0.99891971, -0.02734116],
                  [-0.55983487, 0.001621, -0.82860261],
                  [0.8277518, 0.0464416, -0.55916918]])
	T = np.array([[3.19636], [1.3892], [9.0581]])
	Ri = inv(R)
	Mi = inv(mtx)
	left = Ri.dot(Mi.dot(im))
	right = Ri.dot(T)
	s = (right[2])/(left[2])
	w = s*left - right
	w = w*2.2
	return w


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
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo_6.avi', fourcc, 30.0, (640, 480))
    if cap.isOpened():
        # undistortion matrix
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (640, 480), 1, (640, 480))
        roi_lane = np.array([[0, 480], [640, 480], [640, 250], [0, 250]])
        dst_pts = np.float32([[0, 400], [400, 400], [400, 0], [0, 0]])
        M = cv2.getPerspectiveTransform(np.float32(roi_lane), dst_pts)
        Minv = inv(M)
        i = 0
        while 1:
            ret, frame = cap.read()
            dst = cv2.undistort(frame, camera_matrix, dist_coeff, None, newCameraMatrix)
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            dst = cv2.resize(dst, (640, 480), interpolation=cv2.INTER_NEAREST)
            perspective_img = cv2.warpPerspective(dst, M, (400, 400))
            pp_img = preprocess(perspective_img)
            #cv2.imshow('binary', pp_img)
            loc_mid, end_mid = find_line(pp_img)
            p_loc, p_end = back_perspective(Minv, loc_mid, end_mid)
            loc_line, angle = detect_angle(p_loc, p_end, frame)
			
            if angle > 0:
				angle = 90 - angle
            else:
				angle = -90 - angle
          
            #cv2.putText(loc_line, str(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('frame', loc_line)
            out.write(loc_line)
            # setup PID
            attitude_ctrl = PID.pid(1.08,0,0.5)
            
            # obtain state
            #P_end_i = np.array([xc,yc])
            #P_end_w = Pv.imgpt2wrd(P_end_i)
            #P_loc_i = np.array([415, 859])
            #P_loc_w = Pv.imgpt2wrd(P_loc_i)
            #theta, y = Pv.sensor(P_loc_w, P_end_w)


            # attitude control
            attitude_ctrl.cur = angle
            # attitude_ctrl.desire = 0
            attitude_ctrl.cal_err()
            r_mcd = attitude_ctrl.output()
            RPSR, RPSL = Pv.ctrl_mixer(r_mcd)
            Pv.motor_ctrl(RPSR,RPSL)
            # if angle == 0:

            # else:
      		    # Pv.motor_ctrl_cur(RPSR, RPSL)
            # =============================================================
            # Stop the program on the ESC key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
		out.release()
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == "__main__":
    show_camera()
