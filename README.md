# <center> Capstone Project — JetBot 

## Overview
This project is the review of the course: Capstone Project (F4173), lectured in NCKU, Taiwan. Its objective is to build a embedded self-driving car system, and applied for NVIDIA JetBot eventually. The project is divided into 5 parts, and its final demo.

The key features of the Capstone Project:
+ **Lane Following**
  - Camera Calibration
  - Lane Detection
  - PID Controller
+ **Human Detection**
  - HOG Detection
+ **Traffic Light Detection**
+ **Path Planning**
  - Aruco Marker
  - Digital Map
  - A* Algorithm
+ **Finite State Machine**
  
It's build with:
+ Python 3
+ OpenCV 4
+ Numpy
+ Transitions
---
## Lane Following
This part basically is doing the job to make JetBot detect the traffic lane line, and track it while moving.

### Camera Calibration
First thing is to calibrate the camera. In 3D computer vision, we need to extract metric information from 2D images. In [camera calibration](https://github.com/davidwater/Capstone-Project-JetBot/tree/main/lane%20following/camera%20calibration), using `calibration.py` to find the chessboard corner with different angles, and using `homework.py` to calculate camera matrix, distortion coefficients, rotation and translation vectors.

#### ● Demo
 [![Camera Calibration](http://img.youtube.com/vi/n0G1y1Do7pE/0.jpg)](http://www.youtube.com/watch?v=n0G1y1Do7pE)
 
### Lane Detection
The process of `lane_detection.py` can be separated in 6 steps. Turn the input frame into grayscale, Gaussian blur, canny, image segmentation, and use Hough algorithm to draw the lane.

#### ● Demo
[![Lane Detection](http://img.youtube.com/vi/iLDPHHL6TmU/0.jpg)](http://www.youtube.com/watch?v=iLDPHHL6TmU)
  
### PID Controller
For the purpose of having JetBot drive in the right way, we try to minimize the angle between heading direction and the center of the road. Combining the output of `lane_detection.py` with PID controller to achieve our goal.
+ `PID_1.py`: Function of motor controls
+ `PID_2.py`: Class of PID calculations
+ `PID_final.py`: Main program
  
#### ● Demo
[![Lane Following (first perspective)](http://img.youtube.com/vi/-roYyNna5sg/0.jpg)](http://www.youtube.com/watch?v=-roYyNna5sg)
  
[![Lane Following (third perspective)](http://img.youtube.com/vi/li1O5FhXX_4/0.jpg)](http://www.youtube.com/watch?v=li1O5FhXX_4)
  
***
  
## Human Detection
#### ● Demo
[![Human Detection](http://img.youtube.com/vi/-jEAUBU7DhM/0.jpg)](http://www.youtube.com/watch?v=-jEAUBU7DhM)
  
***
  
## Traffic Light Detection  
#### ● Demo
[![Traffic Light Detection](http://img.youtube.com/vi/rWlAchNVHEs/0.jpg)](http://www.youtube.com/watch?v=rWlAchNVHEs)
  
***
  
## Path Planning
This part is focused on how to let JetBot know state itself and plan a path to the destination.
  
### Aruco Marker
Based on our [detailed map](https://github.com/davidwater/Capstone-Project-JetBot/blob/main/map_detailed.png), we can define our [aruco markers](https://github.com/davidwater/Capstone-Project-JetBot/tree/main/path%20planning/aruco%20marker) as a npz file `aruco_param.npz` which can provide JetBot position and pose on the real-time world.
  
### Digital Map
We create a [digital map](https://github.com/davidwater/Capstone-Project-JetBot/blob/main/path%20planning/digital_map.jpg) to apply for the work of `A* algorithm` afterwards.

### A* Algorithm
A* is a robust algorithm for path planning, and through altering its iteration nodes we can improve the algorithm to use in the real-time application.
+ `A_star_class.py`: Write A* algorithm as a callable class
+ `cubic_spline.py`: Visualize the planning path
+ `path_planning.py`: main program

#### ● Demo
[![Path Planning (backend)](http://img.youtube.com/vi/XfOpXdTvy_g/0.jpg)](http://www.youtube.com/watch?v=XfOpXdTvy_g)
  
[![Path PLanning](http://img.youtube.com/vi/HbOG1u72Ksc/0.jpg)](http://www.youtube.com/watch?v=HbOG1u72Ksc)
  
***

## Finite State Machine
After all relative algorithms are done, our final mission is to integrate all functions to work properly. We used the python transitions package to attain the goal.
+ `modules.py`: all relative classes
+ `fsm.py`: main program
  
#### ● Demo
[![Finite State Machine](http://img.youtube.com/vi/9BevhFvB9FA/0.jpg)](http://www.youtube.com/watch?v=9BevhFvB9FA)
  
***
  
## Final Result
[![Final Review](http://img.youtube.com/vi/MzHDQiKcbdA&t=1s/0.jpg)](http://www.youtube.com/watch?v=MzHDQiKcbdA&t=1s "航太工程實作(二) 第六組 期末回顧影片")
  
