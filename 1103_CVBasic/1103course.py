import cv2
img = cv2.imread("../pic/nick_young.jpg",1)
#print(Mat.cols)
cv2.imshow('show',img)
cv2.waitKey(0)