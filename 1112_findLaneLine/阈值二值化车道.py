import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle



def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size),0)
 
def abs_sobel_thresh(img,orient = 'x',sobel_kernel = 3,thresh = (0,255)):
    gray = grayscale(img)
#用sobel计算梯度
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel))
    
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output
 
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
#根据梯度选取阈值
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separatel
    sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize = sobel_kernel)
    #print(sobel_x)
    sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize = sobel_kernel)
    # 3) Calculate the magnitude 
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(magnitude) / 255
    #print('scale_factor = ',scale_factor)
    magnitude = (magnitude / scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(magnitude)
    # 6) Return this mask as your binary_output image
    binary_output[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1
    return binary_output
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    #print(absgraddir)
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
 
    # Return the binary image
    return binary_output
 
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <thresh[1])] = 1
    return binary_output

def undistort(img):
    cal_pickle = pickle.load(open("../camera_cal_pic/wide_dist_pickle.p", "rb"))
    mtx = cal_pickle['mtx']
    dist = cal_pickle['dist']
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist

'''
img_test = cv2.imread('test_images/straight_lines1.jpg')
undist = undistort(img_test)
x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
color_transforms = hls_select(undist,thresh=(150,255))
color_x_sobel = np.zeros_like(x_sobel)
color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1

cv2.imshow("1", undist)
cv2.imshow("2", x_sobel)
cv2.imshow("3", color_transforms)
cv2.imshow("4", color_x_sobel)

k = cv2.waitKey(0)&0xff

if k == 27:           #按Esc键退出
    cv2.imwrite('./output_images/undist.jpg',undist)
    cv2.imwrite('./output_images/x_sobel.jpg',x_sobel)
    cv2.imwrite('./output_images/color_transforms.png',color_transforms)
    cv2.imwrite('./output_images/color_x_sobel.png',color_x_sobel)
    cv2.destroyAllWindows()
'''

img_test = plt.imread('test_images/straight_lines1.jpg')
plt.figure(figsize = (10,10))
plt.imshow(img_test)
plt.show()
 
undist = undistort(img_test)
plt.subplot(221)
plt.imshow(undist)
plt.title('Undistorted Iamge')
 
cv2.imwrite('./output_images/undist.jpg',undist)
 
x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
plt.subplot(222)
plt.imshow(x_sobel,cmap = 'gray')
plt.title('x_sobel Gradients Image')
 
cv2.imwrite('./output_images/x_sobel.jpg',x_sobel) 
color_transforms = hls_select(undist,thresh=(150,255))
#阈值-二值化
plt.subplot(223)
plt.imshow(color_transforms,cmap = 'gray')
plt.title('Color Thresh Image')
 
cv2.imwrite('./output_images/color_transforms.png',color_transforms)
color_x_sobel = np.zeros_like(x_sobel)
color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1
plt.subplot(224)
plt.imshow(color_x_sobel,cmap = 'gray')
plt.title('color and granient image')
cv2.imwrite('./output_images/color_x_sobel.png',color_x_sobel*255)
plt.show()
cv2.imshow("wtf",undist)
cv2.waitKey(0)
