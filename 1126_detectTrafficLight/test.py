# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:20:44 2018

@author: asus
"""
import matplotlib.image as mpimg
import cv2
import numpy as np
import os
import glob

def standardize_input(image):    # 将所有图片的尺寸全部变换为32x32
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32))
    return standard_im

def crop_image(image):
    row_crop = 4
    col_crop = 7
    return image[row_crop:-row_crop,col_crop:-col_crop,:]

def create_feature(rgb_image):    # 输入RGB图像后，获取输出
    #blur_image = cv2.medianBlur(rgb_image,3)
    croped_image = crop_image(rgb_image)
    hsv = cv2.cvtColor(croped_image, cv2.COLOR_RGB2HSV)
    
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]
        
    red_y = 0
    green_y = 0
    yellow_y = 0
    
    # 统计24x18的图在HSV空间中三个色彩维度在一定范围内的像素点数量
    for i in range(24):    #24列
        red_y += sum( ( (H[i] <= 8) | ((H[i]>=160) & (H[i] <= 180))) & ( S[i] >=43 ) & (V[i] >= 46))
        green_y += sum(( (H[i] >= 85)&(H[i] <= 110) ) & (S[i] >=43) & (V[i] >= 46))
        yellow_y += sum((H[i] <= 25) & (S[i] >= 43) & (V[i] >= 45))      
               
    if((red_y > green_y) and (red_y>yellow_y)):    # 红灯
               return[1,0,0]
    if((green_y>red_y) and (green_y > yellow_y)):    # 绿灯
               return [0,0,1]
    if((yellow_y > red_y) and (yellow_y > green_y)):    # 黄灯
               return [0,1,0]
    return [1,0,0]

def estimate_label(rgb_image):    # 输入RGB图像后，获取对应的预测输出
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = create_feature(rgb_image)
    
    return predicted_label

image_dir = "traffic_light_images/test/"    # 载入测试数据

im = []
for file in glob.glob(os.path.join(image_dir, "try", "*")):
    im.append(mpimg.imread(file))

standard_image=[]
for image in im:
    standard_image.append(standardize_input(image))

#plt.imshow(standard_image)
for image in standard_image:
    predicted_label=estimate_label(image)
    print(predicted_label)

