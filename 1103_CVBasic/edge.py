# -*- coding: cp936 -*-
from cv2 import *
# 分割前边缘检测
img = imread('../pic/road.jpg',0)
edges = Canny(img,100,200)
# 分割后边缘检测
img1 = imread('segment.jpg',0)
edges1 = Canny(img1,100,200)
imshow('origin1',img)
imshow('origin2',img1)
imshow('canny1',edges)
imshow('canny2',edges1)
waitKey(0)