import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt



def show_info(img,left_cur,right_cur,center):
    #在图片中显示出曲率
    cur = (left_cur + right_cur) / 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 使用默认字体
    cv2.putText(img,'Curvature = %d(m)' % cur,(50,50),font,1,(255,255,255),2)
    #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    #添加文字
    
    if center < 0:
        fangxiang = 'left'
    else:
        fangxiang = 'right'
        
    cv2.putText(img,'the angle is %.2fm of %s'%(np.abs(center),fangxiang),(50,100),font,1,(255,255,255),2)

def draw_lines(undist,warped,left_fit,right_fit,left_cur,right_cur,center,show_img = True):
    #创建一个全黑的底层图去划线
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    
    ploty = np.linspace(0,warped.shape[0]-1,warped.shape[0])
    #添加新的多项式在X轴Y轴
    left_fitx = left_fit[0] * ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #把X和Y变成可用的形式
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    #np.transpose 转置
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    #向上/向下翻转阵列。
    pts = np.hstack((pts_left, pts_right))
    #填充图像
    cv2.fillPoly(color_warp, np.int_([pts]), (255,0, 0))
    #透视变换
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    #叠加图层
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    show_info(result, left_cur, right_cur, center)
    if show_img == True:
        plt.figure(figsize = (10,10))
        plt.imshow(result)
        plt.show()
    return result

new_path = os.path.join("test_images/","*.jpg")
for infile in glob.glob(new_path):
    print('the image is ',infile)
    #读图
    img = plt.imread(infile)
    #畸变
    undist = undistort(img)
    #sobel算子
    x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
    #mag_thresh
    mag_binary = mag_thresh(undist,thresh =(30,90))
    #dir_threshold
    dir_binary = dir_threshold(undist, sobel_kernel=15, thresh=(0.7, 1.3))
    #hls颜色阈值
    color_transforms = hls_select(undist,thresh=(150,255))
    #sobel加hls
    color_x_sobel = np.zeros_like(x_sobel)
    color_x_sobel[ (x_sobel == 1) | (color_transforms == 1) ] = 1
    
    #弯曲图像
    warped_img, unpersp, Minv = warp(color_x_sobel)
    #画线
    find_line_imgae,left_fit,right_fit = find_lines(warped_img,print = False)
    #算曲率
    left_curverad, right_curverad, center = curvature(left_fit,right_fit,find_line_imgae,print_data = False)
    #画图
    result = draw_lines(undist,warped_img,left_fit,right_fit,left_curverad,right_curverad,center)

