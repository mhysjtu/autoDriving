import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)#构建一个72行，3列的零矩阵
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)#把数组变成网格的顺序
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
 
# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    print('number:',fname,'ret = ',ret)
 
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

img = cv2.imread('camera_cal/calibration1.jpg')
print(img.shape)
img_size = (img.shape[1],img.shape[0])
print(img_size)
# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)#标定
#这个函数会返回标定结果、相机的内参数矩阵、畸变系数、旋转矩阵和平移向量。
 
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

def undistort(img):
    cal_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = cal_pickle['mtx']
    dist = cal_pickle['dist']
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist

image_test = 'camera_cal/calibration3.jpg'  #TODO:换成其他图片试一试
img_test = cv2.imread(image_test)
img_undistort = undistort(img_test)
 
plt.figure(figsize = (15,15))
plt.subplot(121)   #绘制一行两列的子图
plt.imshow(img_test)
plt.title('Original image')
 
plt.subplot(122)
plt.imshow(img_undistort)
plt.title('Undistort image')
dist
