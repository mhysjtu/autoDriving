import matplotlib.pyplot as plt
import cv2

image_num = 800    # 以一个图像为例
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]
 
# Convert to HSV
hsv = cv2.cvtColor(test_im, cv2.COLOR_BGR2HSV)    # 色彩空间变换 RGB->HSV
# Print image label
print('Label [red, yellow, green]: ' + str(test_label))
 
# HSV channels
h = hsv[:,:,0]    # 色调
s = hsv[:,:,1]    # 饱和度
v = hsv[:,:,2]    # 明度
# Plot the original image and the three channels
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,10))
ax1.set_title('Standardized image')
ax1.imshow(hsv)
ax2.set_title('H channel')
ax2.imshow(h,cmap = 'gray')
ax3.set_title('S channel')
ax3.imshow(s, cmap='gray')
ax4.set_title('V channel')
ax4.imshow(v, cmap='gray')

# 剪掉图像里最边上的像素，将32x32图像裁剪成24x18
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