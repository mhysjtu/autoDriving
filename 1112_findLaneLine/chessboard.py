# -*- coding: cp936 -*-
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp =np.zeros((6*9,3),np.float32)#����һ��72�У�3�е������
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)#�������������˳��
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
 
# Make a list of calibration images
images = glob.glob('../camera_cal_pic/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Find the chessboard corners
    ret, corners =cv2.findChessboardCorners(gray, (9,6), None)
    print('number:',fname,'ret = ',ret)
 
    # If found, add object points, image points
    if ret == True:
           objpoints.append(objp)
           imgpoints.append(corners)
 
        # Draw and display the corners
           cv2.drawChessboardCorners(img, (9,6), corners, ret)
           plt.figure(figsize = (8,8))
           plt.imshow(img)
           plt.show()