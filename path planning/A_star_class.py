import cv2
import numpy as np
import cubic_spline
from robot import Robot
import time
import timeit

robot = Robot()

class AStar():
    def __init__(self, m):
        self.map = m
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {}  # Distance from start to node
        self.g = {}  # Distance from node to goal
        self.goal_node = None

    # estimation
    def _distance(self, a, b):
        # Diagonal distance
        d = np.max([np.abs(a[0]-b[0]), np.abs(a[1]-b[1])])
        return d

    def planning(self, start, goal, inter=20, img=None):
        # Initialize
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = self._distance(start, goal)
        node_goal = None
        while(1):
            min_dist = 99999
            min_id = -1
            for i, node in enumerate(self.queue):
                # i, node = 1, (100, 200)

                # todo
                #####################################################
                # In a*  we need to add something in this function
                # g(v) 已經弄好了，要多求 h(v)

                f = self.g[node]  # find distance from start to this node
                y = self.h[node]  # find distance from node to goal

                #####################################################
                if f+y < min_dist:
                    min_dist = f+y
                    min_id = i

            # pop the nearest node
            p = self.queue.pop(min_id)

            # meet obstacle, skip
            if self.map[p[1], p[0]] < 0.5:
                continue

            # find goal
            if self._distance(p, goal) < inter:
                self.goal_node = p
                break

            # eight direction
            pts_next1 = [(p[0]+inter, p[1]), (p[0], p[1]+inter),
                         (p[0]-inter, p[1]), (p[0], p[1]-inter)]
            pts_next2 = [(p[0]+inter, p[1]+inter), (p[0]-inter, p[1]+inter),
                         (p[0]-inter, p[1]-inter), (p[0]+inter, p[1]-inter)]
            pts_next = pts_next1 + pts_next2

            for pn in pts_next:
                # 下個點不在 parent 裡面
                if pn not in self.parent:
                    self.queue.append(pn)
                    self.parent[pn] = p # store next point's parent is current p
                    self.g[pn] = self.g[p] + inter # estimation of next point g(v)

                    # todo
                    ##############################################
                    # update the estimation of h(v)
                    self.h[pn] = self._distance(pn, goal)

                    ##############################################
                elif self.g[pn] > self.g[p] + inter:
                    self.parent[pn] = p
                    self.g[pn] = self.g[p] + inter

            if img is not None:
                cv2.circle(img, (start[0], start[1]), 5, (0, 0, 1), 3)
                cv2.circle(img, (goal[0], goal[1]), 5, (0, 1, 0), 3)
                cv2.circle(img, p, 2, (0, 0, 1), 1)
                img_ = cv2.flip(img, 0)
                cv2.imshow("A* Test", img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break

        # Extract path
        path = []
        p = self.goal_node
        while(True):
            path.insert(0, p)
            if self.parent[p] == None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path

    def decide_direction(self, p_0, p_1, att):
        up = 1
        right = 2
        down = 3
        left = 4

        if p_1[0] > p_0[0] and att == right:  # head for the right && toward right
            robot.forward(0.15) # go straight
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[0] > p_0[0] and att == left:  # head for the right && toward left
            robot.set_motors(0.1, 0.2)  # turn left
            time.sleep(2)
            planning = True
            pass
        else:
            planning = False

        if p_1[0] > p_0[0] and att == up:  # head for the right && toward upward
            robot.set_motors(0.2, 0.1)  # turn right
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[0] > p_0[0] and att == left:  # head for the left && toward left
            robot.forward(0.15)  # go straight
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[1] > p_0[1] and att == left:   # head for the left && toward left
            robot.set_motors(0.1, 0.2)  # turn left
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[0] > p_0[0] and att == down:  # head for right && toward downward
            robot.set_motors(0.1, 0.2)  # turn left
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[1] > p_0[1] and att == down:  # head for downward && toward downward
            robot.forward(0.15)  # go straight
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        if p_1[1] > p_0[1] and att == right: # head for downward && toward right
            robot.set_motors(0.2, 0.1)  # turn right
            time.sleep(1)
            planning = True
            pass
        else:
            planning = False

        return planning


    def aruco_marker(self):
        return True

