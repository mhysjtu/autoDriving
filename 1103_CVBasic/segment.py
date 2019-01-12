# -*- coding: cp936 -*-
import cv2
import numpy as np
import time 
from matplotlib import pyplot as plt

# manual experience人工经验
thresh=[0, 107]
# img =cv2.imread('pic/road.jpg',cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img,(5,5),2.0)
# print(type(img))
# binary = np.zeros_like(img)
# binary[(img <= thresh[1])&(img >=thresh[0])]=255

# cv2.imshow('binary',binary)

# cv2.waitKey(0)

img = cv2.imread('../pic/road.jpg',-1)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray1,(5,5),2.0)
#hist = cv2.calcHist([gray],[0],None,[255],[0,255])
#plt.hist(img.ravel(),256,[0,256])
#plt.show()
time0 = time.time()
retval, dst = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
time1 = time.time()
total = (time1-time0)
print("otsu need time: %.3f s" % total)
dst1 = 255-dst
cv2.imwrite('segment.jpg',dst)

binary = np.zeros_like(gray)
binary[(gray <= thresh[1])&(gray >=thresh[0])]=255

cv2.imshow('binary',binary)
cv2.imshow("src",img)
cv2.imshow("gray",gray)
cv2.imshow("dst",dst)
cv2.waitKey(0)