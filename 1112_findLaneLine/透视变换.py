import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("stopsign.jpg")
plt.imshow(img)

print(img.shape)
#TODO:根据实际图片找点
plt.plot(494,81,'x')  #右上角
plt.plot(517,155,'x')  #右下角
plt.plot(270,239,'x')  #左下角
plt.plot(278,167,'x')  #左上角

def warp(img):
    img_size = (img.shape[1],img.shape[0])
    src = np.float32( [ [494,81],[517,155],[270,239],[278,167]] )
    dst = np.float32( [ [510,70],[510,160],[270,160],[270,70]] )
    M = cv2.getPerspectiveTransform(src,dst)
    #返回透视变换的映射矩阵，就是这里的M
    #对于投影变换，我们则需要知道四个点，
    #通过cv2.getPerspectiveTransform求得变换矩阵.之后使用cv2.warpPerspective获得矫正后的图片。
    Minv = cv2.getPerspectiveTransform(dst,src)  #反变换所需的矩阵Minv
    warped = cv2.warpPerspective(img,M,img_size,flags = cv2.INTER_LINEAR)
    #主要作用：对图像进行透视变换，就是变形
    return warped
warped_img = warp(img)
plt.figure(figsize = (15,15))
plt.subplot(121)   #绘制一行两列的子图
plt.imshow(warped_img)
plt.title('warped image')
plt.subplot(122)
plt.imshow(img)
plt.title('original image')
