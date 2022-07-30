import A_star_class as astar
from jetbot import Robot
import cv2
import numpy as np
import timeit


robot = Robot()
smooth = True
while 1:
    robot.stop()
    img = cv2.flip(cv2.imread("map3_v3.jpg"), 0)
    img[img > 128] = 255
    img[img <= 128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m = 1-cv2.dilate(1-m, np.ones((20, 20)))
    img = img.astype(float)/255.

    file = np.load('aruco_param.npz')
    parameter = file['parameter']
    id = int(input())
    face = parameter[:, 1]
    coordinate = parameter[:, 2:3]
    face1 = face[id]
    coor = coordinate[id]
    start = (coor[0], coor[1])
    goal = (570, 20)

    a = timeit.default_timer()
    astar = AStar(m)
    path = astar.planning(start=start, goal=goal, img=img, inter=21)
    print(path)

    # determine Jetbot direction
    p_0 = path[0]
    p_1 = path[1]

    planning = decide_direction(p_0, p_1, face1)

    b = timeit.default_timer()
    print("Time: ", b-a)


    cv2.circle(img, (start[0], start[1]), 5, (0, 0, 1), 3)
    cv2.circle(img, (goal[0], goal[1]), 5, (0, 1, 0), 3)

    # Extract Path
    if not smooth:
        for i in range(len(path)-1):
            cv2.line(img, path[i], path[i+1], (1, 0, 0), 2)
    else:
        path = np.array(cubic_spline.cubic_spline_2d(path, interval=1))
        for i in range(len(path)-1):
            cv2.line(img, cubic_spline.pos_int(path[i]), cubic_spline.pos_int(path[i+1]), (1, 0, 0), 1)

    img_ = cv2.flip(img, 0)
    cv2.imshow("A* Test", img_)
    k = cv2.waitKey(0)

