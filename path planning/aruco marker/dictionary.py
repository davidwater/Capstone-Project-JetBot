import numpy as np

id = np.arange(22)
face = np.array([3, 4, 1, 2, 1, 4, 2, 3, 2, 4, 2, 3, 4, 3, 1, 2, 4, 1, 3, 1, 3, 4])
# coordinate = [(10, 140), "", "", (130, 30), "", "", (130, 170), (140, 270), (310, 120)
# , "", (310, 255), (270, 270), "", (140, 140), (270, 170), "", ""
# , (270, 140), "", (10, 0), (310, 300)]
coordinate = np.array([[25,290], [35,550], [32,550], [290,550], [25,290], [290,550],
                       [300,290], [290,30], [290,30], [290,30],
                       [530,40], [530,40], [25,290], [300,290],
                       [300,290], [560,290], [300,290], [290,550], [560,290], [560,290],
                       [32,550], [570, 20]])


parameter = np.transpose(np.vstack([id, face]))
parameter = np.hstack([parameter, coordinate])
print(parameter)
np.savez('aruco_param.npz', parameter = parameter)
ep = np.load('aruco_param.npz')
ep = ep['parameter']
print(ep)

