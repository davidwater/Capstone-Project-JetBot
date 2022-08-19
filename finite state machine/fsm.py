from transitions import Machine
import time
import cv2
import numpy as np
from modules import detection
from lane_detect import lane
from A_star_class import AStar
import PID_v2_1 as Pv
import PID_v2_2 as PID
import cubic_spline
import timeit
from robot import Robot
import random

class FSM(object):
    # define states
    states = ['path_planning', 'traffic_light', 'openloop_motion', 'lane_detection', 'avoid_people']
    # main code
    def __init__(self, robot):
        self.robot = robot
        self.machine = Machine(model=self, states=FSM.states, initial='path_planning')
        self.machine.add_transition(trigger='find_path', source='path_planning', dest='traffic_light', before='stop_jetbot')
        self.machine.add_transition(trigger='green_light', source='traffic_light', dest='openloop_motion', before='stop_jetbot')
        self.machine.add_transition(trigger='pass_intersection', source='openloop_motion', dest='lane_detection')
        self.machine.add_transition(trigger='stop_line', source='lane_detection', dest='path_planning', after='stop_jetbot')
        self.machine.add_transition(trigger='detect_people', source='*', dest='avoid_people')
        self.machine.add_transition(trigger='avoid_success', source='avoid_people', dest='lane_detection', after='stop_jetbot')
        

    def stop_jetbot(self):
        r = self.robot
        r.set_motors(0, 0)
        time.sleep(1)

def show_state(fsm):
    print('current state: ', fsm.state)

robot = Robot()
cap = cv2.VideoCapture(detection.gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('0525.avi', fourcc, 30.0, (640, 480))
file = np.load('calibration.npz')
file1 = np.load('aruco_param.npz')
fsm = FSM(robot)
# define parameter
direction = None
green_light = True
current_id = 8
id_distance = None
detect_people = None
people = 0
traffic_time = 0
while(1):
    ret, frame = cap.read()
    d = detection(frame, file)
    state = fsm.state

    # detect aruco
    undistorted_image = d.undistortion(frame)
    gray = d.preprocess(frame)
    parameter = d.aruco_detect(gray)
#     print('parameter: ', parameter)
    if parameter is not None:
        current_id_fake = int(parameter[0])
        id_distance = parameter[1]
    
    
    # detect people
    if people == 0:
        detect_people = False
        detect_people = d.HOG_detect()
        print('detect people: ', detect_people)
        if detect_people == True:
            fsm.detect_people()
            people += 1
    
    booll = [True, False]
    n = random.randint(0, 1)
    # show current FSM state
    show_state(fsm)

    # define 
    if state == 'path_planning':
        print('-----------------------')
        print('start path planning....')
        smooth = True

        #     digital map preprocessing
        img = cv2.flip(cv2.imread("map_v5.jpg"), 0)
        img[img > 128] = 255
        img[img <= 128] = 0
        m = np.asarray(img)
        m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        m = m.astype(float) / 255.
        m = 1 - cv2.dilate(1 - m, np.ones((20, 20)))
        img = img.astype(float) / 255.

        #     Aruco marker input
        parameter = file1['parameter']
        print('current_id: ', current_id)
        id = current_id 
        face = parameter[:, 1]
        coordinate = parameter[:, 2:4]
        face1 = face[id]
        coor = coordinate[id]
        start = (coor[0], coor[1])
        goal = (570, 20)

        a = timeit.default_timer()
        astar = AStar(m)
        path = astar.planning(start=start, goal=goal, img=img, inter=22)
        print('path[0]: ',path[0])
        print('path[1]: ',path[1])

        # determine Jetbot direction
        p_0 = path[0]
        p_1 = path[1]

        planning, dir = astar.decide_direction(p_0, p_1, face1)
        print('direction:', dir)

        b = timeit.default_timer()
        print("Time: ", b - a)

        cv2.circle(img, (start[0], start[1]), 5, (0, 0, 1), 3)
        cv2.circle(img, (goal[0], goal[1]), 5, (0, 1, 0), 3)

        # Extract Path
        if not smooth:
            for i in range(len(path) - 1):
                cv2.line(img, path[i], path[i + 1], (1, 0, 0), 2)
        else:
            path = np.array(cubic_spline.cubic_spline_2d(path, interval=1))
            for i in range(len(path) - 1):
                cv2.line(img, cubic_spline.pos_int(path[i]), cubic_spline.pos_int(path[i + 1]), (1, 0, 0), 1)

        img_ = cv2.flip(img, 0)
        fsm.find_path()

    elif state == 'traffic_light':
        print('-----------------------')
        print('start traffic light detection....')
        if traffic_time == 0:
            green_light = d.traffic_light_detect()
            if green_light == True:
                traffic_time += 1
        else:
            green_light = True
            
        if green_light == True:
            fsm.green_light()
        print('green light: ', green_light)

    elif state == 'openloop_motion':
        print('-----------------------')
        print('start open loop motion....')
        if dir == 'left':
            robot.set_motors(0.11,0.2)
            time.sleep(2.7)
            robot.set_motors(0.101,0.108)
            time.sleep(2)
            robot.set_motors(0, 0)
            print('turning left....')
        elif dir == 'right':
            robot.set_motors(0.154,0.15)
            time.sleep(0.15)
            robot.set_motors(0.215, 0.1)
            time.sleep(1.5)
            robot.set_motors(0.101,0.108)
            time.sleep(2)
            robot.set_motors(0, 0)
            print('turning right....')
        else:
            robot.set_motors(0.125,0.14)
            time.sleep(3)
            robot.set_motors(0, 0)
            print('go straight....')
        fsm.pass_intersection()

    elif state == 'avoid_people':
        print('-----------------------')
        print('start avoid people....')
        d.avoid_people(robot)
        fsm.avoid_success()
    else:
        print('-----------------------')
        print('start lane detection....')
        stop = False
        stop = d.stop_line_detect(800)
#         if id_distance is not None:
#             if id_distance < 50:
#                 stop = True
        print('detect stop line: ', stop)
        print('id idstance: ', id_distance)
        if stop == True:
            fsm.stop_line()
        else:
            # lane tracking
            l = lane(frame)
            roi_image, mask = l.preprocess()
            angle, start, end = l.lane_calculate(mask, roi_image)
            if angle is not None:
#                 if angle > 5:
#                     angle = 3
#                     print('angle = ', angle)
#                 elif angle < -5:
#                     angle = -2
#                     print('angle = ', angle)
                print('angle = ', angle)
                # setup PID
                attitude_ctrl = PID.pid(0.89, 0, 0)
                # attitude control
                attitude_ctrl.cur = - angle
                # attitude_ctrl.desire = 0
                attitude_ctrl.cal_err()
                r_mcd = attitude_ctrl.output()
                RPSR, RPSL = Pv.ctrl_mixer(r_mcd)
                left_V, right_V = Pv.motor_ctrl(RPSR, RPSL)
                print(left_V)
                print(right_V)
                robot.set_motors(left_V+0.01, right_V+0.01)
#     cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cap.release()
out.release()
# cv2.destroyAllWindows()