from cv2 import *
import numpy as np
# 1 for Probabilistic_Hough_Transform
# 0 for Classic_Hough_Transform
flag = 0
img = imread('../pic/line.jpg')
gray = cvtColor(img, COLOR_BGR2GRAY)
edges = Canny(gray, 100 ,200, apertureSize = 3)

if flag == 0:
	lines = HoughLines(edges, 1, np.pi/180,150)
	lines1 = lines[:,0,:]
	for rho, theta in lines1[:]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0+1000*(-b))
		y1 = int(y0+1000*(a))
		x2 = int(x0-1000*(-b))
		y2 = int(y0-1000*(a))
		line(img,(x1,y1),(x2,y2),(0,255,0),2)
else:
	lines = HoughLinesP(edges, 1, np.pi/180,80,50,15)
	lines1 = lines[:,0,:]
	for x1,y1,x2,y2 in lines1[:]:
		line(img,(x1,y1),(x2,y2),(0,255,0),2)

imshow('lines',img)
waitKey(0)