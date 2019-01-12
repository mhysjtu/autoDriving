import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("../pic/toushi.jpg")
plt.imshow(img)

print(img.shape)

plt.plot(494,81,'x')
plt.plot(517,155,'x')
plt.plot(270,239,'x')
plt.plot(278,167,'x')

def warp(img):
	img_size = (img.shape[1], img.shape[0])
	src = np.float32([[494,81],[517,155],[270,239],[278,167]])
	#dst = np.float32([[510,70],[510,160],[270,160],[270,70]])
	dst = np.float32([[620,180],[620,270],[380,270],[380,180]])
	M = cv2.getPerspectiveTransform(src,dst)
	
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(img,M,img_size,flags = cv2.INTER_LINEAR)
	return warped
	
warped_img = warp(img)
plt.figure(figsize = (15,15))
plt.subplot(121)
plt.imshow(warped_img)
plt.title('warped image')
plt.subplot(122)
plt.imshow(img)
plt.title('origin image')
plt.show()

