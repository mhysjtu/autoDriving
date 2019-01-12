import cv2
import random
import numpy as np
import test_functions

def standardize_input(image):    # 将所有图片的尺寸全部变换为32x32
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im,(32,32))
    return standard_im

def one_hot_encode(label):    # 输出设置三个维度（红黄绿灯），将图像对应的输出维度置1，其余置0
    ## TODO: Create a one-hot encoded label that works for all classes of traffic lights
#    one_hot_encoded = []    # 这个列表有用吗？？？？？？   没用
    if label == "red":
        return [1,0,0]
    elif label == "yellow":
        return [0,1,0]
    else:
        return [0,0,1]

def standardize(image_list):    # 输入和输出的标准化
    
    # Empty image data array
    standard_list = []
 
    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]    # 图像
        label = item[1]    # 标签
 
        # Standardize the image
        standardized_im = standardize_input(image)    # 将图像标准化（32x32）
 
        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    # 输出组合
 
        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))    # 加入列表中
        
    return standard_list

tests = test_functions.Tests()
 
# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)    # 所有标准化的图像和标签（label）

standard_image = STANDARDIZED_LIST[525][0]    # 挑一张图片输出并显示信息
print(standard_image.shape)
plt.title(STANDARDIZED_LIST[0][1])
plt.imshow(standard_image)
