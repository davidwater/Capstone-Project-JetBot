import numpy as np
import cv2
from numpy.linalg import inv, norm
from math import acos, pi, atan

class lane:
    def __init__(self, frame) :
        self.frame = frame
        self.perspective_frame = None
        self.mask = None
        self.roi_lane = np.array([[0, 480], [640, 480], [640, 250], [0, 250]])
        self.dst_pts = np.float32([[0, 400], [400, 400], [400, 0], [0, 0]])

    def preprocess(self):
        frame = self.frame
        roi_lane = self.roi_lane
        dst_pts = self.dst_pts
        M = cv2.getPerspectiveTransform(np.float32(roi_lane), np.float32(dst_pts))
        roi_image = cv2.warpPerspective(frame, M, (400, 400))
        self.perspective_frame = roi_image
        # split into left and right
        right = roi_image[:, 200:400]
        left = roi_image[:, 0:200]
        hsv = cv2.cvtColor(left, cv2.COLOR_BGR2HSV)
        # real scenario
        lower = np.array([0, 24, 170])
        upper = np.array([151, 137, 255])
        # right part
        gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
        # home testing scenario
        # lower = np.array([0, 10, 150])
        # upper = np.array([255, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower, upper)

        mask = np.hstack([mask_hsv, thresh])
        self.mask = mask
        return roi_image, mask
    
    def detect_angle(left_fit, right_fit):
        y_start = 150
        y_end = 400
        # start
        right_x_start = right_fit[0] * y_start **2 + right_fit[1] * y_start + right_fit[2]
        left_x_start = left_fit[0] * y_start ** 2 + left_fit[1] * y_start + left_fit[2]
        mid_x_start = int((right_x_start + left_x_start)/2)
        # end
        right_x_end = right_fit[0] * y_end ** 2 + right_fit[1] * y_end + right_fit[2]
        left_x_end = left_fit[0] * y_end **2 + left_fit[1] * y_end + left_fit[2]
#         mid_x_end = int((right_x_end + left_x_end)/2)
        mid_x_end = 216
        # calculate angle
        dx = mid_x_start - mid_x_end
        dy = y_end - y_start
        angle = atan(dx/dy) * 180/pi
        return angle, (mid_x_start, y_start), (mid_x_end, y_end)

    def single_lane_detect(fit, side):
        y_start = 150
        y_end = 400
        x_start = fit[0]*y_start**2 + fit[1]*y_start + fit[2]
        x_end = fit[0]*y_end**2 + fit[1]*y_end + fit[2]
        if x_start <= x_end:
            # only right lane
#             mid_x_end = x_end - 153
            mid_x_end = 216
            mid_x_start = x_start - 103
        else:
            # only left lane
            mid_x_end = 216
            mid_x_start = x_start + 103

        # calculate angle
        dx = mid_x_start - mid_x_end
        dy = y_end - y_start
        angle = atan(dx/dy) * 180/pi
        return angle, (int(mid_x_start), y_start), (int(mid_x_end), y_end)


    def draw_line(left_fit=None, right_fit=None):
        y = np.arange(400)
        xl = np.zeros_like(y)
        xr = np.zeros_like(y)
        if left_fit is not None:
            for i in range(400):
                yi = y[i]
                xi = left_fit[0]*yi**2 + left_fit[1]*yi + left_fit[2]
                xl[i] = int(xi)
        if right_fit is not None:
            for j in range(400):
                yj = y[j]
                xj = right_fit[0]*yj**2 + right_fit[1]*yj + right_fit[2]
                xr[j] = int(xj)
        left_pts = np.transpose(np.vstack([xl, y]))
        right_pts = np.transpose(np.vstack([xr, y]))
        
        return left_pts, right_pts

    def lane_calculate(self, perspective_mask, perspective_image):
        mask = perspective_mask
        hist = np.sum(mask, axis=0)
        peak_left = np.argmax(hist[0:200])
        peak_right = np.argmax(hist[200:400]) + 200
        n_window = 20
        h_window = int(400/n_window)
        w_window = 50
        left_current_x = peak_left
        right_current_x = peak_right
        min_pixels = 50
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_ids = []
        right_ids = []
        mid_point = []
        for n in range(n_window):
            # window range
            y_min = 400 - n * h_window
            y_max = y_min + h_window
            x_min_l = left_current_x - w_window
            x_max_l = left_current_x + w_window
            x_min_r = right_current_x - w_window
            x_max_r = right_current_x + w_window
            # find nonzero value in window
            left_nonzeros_ids = ((nonzeroy >= y_min) & (nonzeroy < y_max) & 
                                (nonzerox >= x_min_l) & (nonzerox < x_max_l)).nonzero()[0]
            right_nonzeros_ids = ((nonzeroy >= y_min) & (nonzeroy < y_max) &
                                (nonzerox >= x_min_r) & (nonzerox < x_max_r)).nonzero()[0]
            left_ids.append(left_nonzeros_ids)
            right_ids.append(right_nonzeros_ids)
            # lane mid x
            mid_x = int((left_current_x + right_current_x)/2)
            mid_point.append([mid_x, y_min])
#             cv2.circle(perspective_image, (mid_x, y_max), 5, (255, 255, 0), -1)
            # update left & right current x
            if len(left_nonzeros_ids) > min_pixels:
                left_current_x = int(np.mean(nonzerox[left_nonzeros_ids]))
            if len(right_nonzeros_ids) > min_pixels:
                right_current_x = int(np.mean(nonzerox[right_nonzeros_ids]))
            # cv2.rectangle(perspective_image, (x_min_l , y_min), (x_max_l, y_max), (255, 0, 0), 2)
            # cv2.rectangle(perspective_image, (x_min_r, y_min), (x_max_r, y_max), (255, 0, 0), 2)
        left_ids = np.concatenate(left_ids)
        right_ids = np.concatenate(right_ids)
        leftx = nonzerox[left_ids]
        lefty = nonzeroy[left_ids]
        rightx = nonzerox[right_ids]
        righty = nonzeroy[right_ids]
        thresh = 5000
        # print('left', np.mean(leftx), 'right', np.mean(rightx))
        
        # identify left lane and right lane
        if (len(leftx) > thresh) & (np.mean(leftx) <= 199):
            left = True
        else:
            left = False
        if (len(rightx) > thresh) & (np.mean(rightx) >= 200):
            right = True
        else:
            right =  False
        # four conditions
        if (left == True) & (right == True):
            # print('left + right')
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            angle_lane, start, end = lane.detect_angle(left_fit, right_fit)
            left_pts, right_pts = lane.draw_line(left_fit, right_fit)
            cv2.polylines(perspective_image, [right_pts], False, (0, 0, 255), 5)
            cv2.polylines(perspective_image, [left_pts], False, (255, 0, 0), 3)
        
        elif (left == True) & (right == False):
            # print('left')
            left_fit = np.polyfit(lefty, leftx, 2)
            y = np.arange(400)
            xl = np.zeros_like(y)
            if left_fit is not None:
                for j in range(400):
                    yj = y[j]
                    xj = left_fit[0]*yj**2 + left_fit[1]*yj + left_fit[2]
                    xl[j] = int(xj)
            left_pts = np.transpose(np.vstack([xl, y]))
            cv2.polylines(perspective_image, [left_pts], False, (255, 0, 0), 3)
            angle_lane, start, end = lane.single_lane_detect(left_fit, 'left')

        elif (left == False) & (right == True):
            # print('right')
            right_fit = np.polyfit(righty, rightx, 2)
            y = np.arange(400)
            xr = np.zeros_like(y)
            if right_fit is not None:
                for j in range(400):
                    yj = y[j]
                    xj = right_fit[0]*yj**2 + right_fit[1]*yj + right_fit[2]
                    xr[j] = int(xj)
            right_pts = np.transpose(np.vstack([xr, y]))
            cv2.polylines(perspective_image, [right_pts], False, (0, 0, 255), 5)
            angle_lane, start, end = lane.single_lane_detect(right_fit, 'right')
        else:
            # print('Both none')
            angle_lane = None
            start = None
            end = None
            pass

        return angle_lane, start, end