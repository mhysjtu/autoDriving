import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

color_x_sobel = plt.imread('output_images/color_x_sobel.png')
plt.imshow(color_x_sobel,cmap = 'gray')
print(color_x_sobel.shape)
plt.plot(800,510,'x')
plt.plot(1150,700,'x')
plt.plot(270,700,'x')
plt.plot(510,510,'x')
 
'''
plt.plot(650,470,'x')
plt.plot(640,700,'x')
plt.plot(270,700,'x')
plt.plot(270,520,'x')
'''
def warp(img):
    img_size = (img.shape[1],img.shape[0])
    
    src = np.float32( [ [800,510],[1150,700],[270,700],[510,510]] )
    dst = np.float32( [ [650,470],[640,700],[270,700],[270,540]] )
    M = cv2.getPerspectiveTransform(src,dst)
    #返回透视变换的映射矩阵，就是这里的M
    #对于投影变换，我们则需要知道四个点，
    #通过cv2.getPerspectiveTransform求得变换矩阵.之后使用cv2.warpPerspective获得矫正后的图片。
    
    Minv = cv2.getPerspectiveTransform(dst,src)
    
    warped = cv2.warpPerspective(img,M,img_size,flags = cv2.INTER_LINEAR)
    #主要作用：对图像进行透视变换，就是变形
    #https://blog.csdn.net/qq_18343569/article/details/47953843
    unpersp = cv2.warpPerspective(warped, Minv, img_size, flags=cv2.INTER_LINEAR)
    
    
    return warped, unpersp, Minv

warped_img,unpersp, Minv = warp(color_x_sobel)

plt.figure(figsize = (15,15))
plt.subplot(121)   #绘制一行两列的子图
plt.imshow(warped_img,cmap = 'gray')
plt.title('warped image')
cv2.imwrite('./output_images/warped_img.jpg',warped_img*255)
 
plt.subplot(122)
plt.imshow(unpersp,cmap = 'gray')
plt.title('original image')
plt.show()


