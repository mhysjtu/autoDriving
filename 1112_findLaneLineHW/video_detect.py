# video car line detection
# 问题：车道透视变换时要人工给出四个点，这在视频流中怎么处理？
#       判断是否为第一帧，若是。。。；若不是根据上一帧做处理，怎么根据上一帧处理？

# import libraries
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

# function definition
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

def warp(img):
    img_size = (img.shape[1],img.shape[0])
	
	# 640*368
    src = np.float32([[360,240],[510,300],[210,300],[280,240]])
    dst = np.float32([[450,120],[450,360],[180,360],[180,120]])

	# 1280*720
    # src = np.float32( [ [800,510],[1150,700],[270,700],[510,510]] )
    # dst = np.float32( [ [650,470],[640,700],[270,700],[270,540]] )
	
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

def find_lines(img,print = True):
    #假设您已经创建了一个被扭曲的二进制图像，称为“binary_warped”
    #取图像下半部分的直方图
    histogram= np.sum(img[img.shape[0] //2:,:],axis = 0)
    #创建一个输出图像来绘制和可视化结果
    out_img = np.dstack((img,img,img))*255
    # plt.imshow(out_img)
    # plt.show()
    #找出直方图的左半边和右半边的峰值
    #这些将是左行和右行的起点
    midpoint = np.int(histogram.shape[0] // 4)
    leftx_base = np.argmax(histogram[:midpoint])
    #np.argmax 是返回最大值所在的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #这里是要返回右边HOG值最大所在的位置，所以要加上midpoint
 
    #选择滑动窗口的数量
    nwindows = 9
    #设置窗口的高度
    window_height = np.int(img.shape[0] // nwindows)
    #确定所有的x和y位置非零像素在图像,这里就是吧img图像中非0元素（就是不是黑的地方就找出来，一行是x，一行是y）
    nonzero = img.nonzero()
    #返回numpy数组中非零的元素
    #对于二维数组b2，nonzero(b2)所得到的是一个长度为2的元组。http://www.cnblogs.com/1zhk/articles/4782812.html
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #为每个窗口当前位置更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    #设置窗口的宽度+ / -
	
    # #1280*640
	# margin = 100
	
	#640*368
    margin = 50
	
    #设置最小数量的像素发现重定位窗口
	
	# #1280*640
    # minpix = 50
	
	#640*368
    minpix = 10
	
    #创建空的列表接收左和右车道像素指数
    left_lane_inds = []
    right_lane_inds = []
 
    #遍历窗口
    for window in range(nwindows):
        #识别窗口边界在x和y(左、右)
        win_y_low = img.shape[0] - (window + 1) * window_height #就是把图像切成9分，一分一分的算HOG
        #print('win_y_low',win_y_low)
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        #print('win_xleft_low',win_xleft_low)
        win_xleft_high = leftx_current + margin
        #print('win_xleft_high = ',win_xleft_high)
        win_xright_low = rightx_current - margin
        #print('win_xright_low = ',win_xright_low)
        win_xright_high = rightx_current + margin
        #print('win_xright_high = ',win_xright_high)
        #把网格画在可视化图像上
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)#通过确定对角线 画矩形
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
 
    #     plt.imshow(out_img)
    #     plt.show()
    #     print('left !!!! ',win_xleft_low,win_y_low,win_xleft_high,win_y_high)
    #     print('right !!!!! ',win_xright_low,win_y_low,win_xright_high,win_y_high)
 
        #识别非零像素窗口内的x和y
        good_left_inds = (  (nonzeroy >= win_y_low)  & (nonzeroy < win_y_high)  
                              & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
 
 
        good_right_inds = ( (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                              & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
 
        #添加这些指标列表
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #如果上面大于minpix，重新定位下一个窗口的平均位置
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #连接索引的数组
    left_lane_inds = np.concatenate(left_lane_inds)
    #把list改成numpy格式而已
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #提取左和右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    #最小二乘多项式拟合。
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    #画图
    ploty = np.linspace(0,img.shape[0] -1,img.shape[0]) #用此来创建等差数列
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty +left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 +right_fit[1] * ploty + right_fit[2]
    #这步的意思是把曲线拟合出来，
 
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    if print == True:
        plt.figure(figsize=(8,8))
        
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.show()
    
    return out_img,left_fit,right_fit	
	
def curvature(left_fit,right_fit,binary_warped,print_data = True):
    ploty = np.linspace(0,binary_warped.shape[0] -1 , binary_warped.shape[0])
    y_eval = np.max(ploty)
    #y_eval就是曲率，这里是选择最大的曲率
    
	#1280*720
    # ym_per_pix = 30/720#在y维度上 米/像素
    # xm_per_pix = 3.7/700#在x维度上 米/像素
	
	#640*368
    ym_per_pix = 30/368#在y维度上 米/像素
    xm_per_pix = 3.7/368#在x维度上 米/像素
    
    #确定左右车道
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #定义新的系数在米
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    #最小二乘法拟合
    
    #计算新的曲率半径
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    #计算中心点，线的中点是左右线底部的中间
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
	
	#1280*720
    #center_image = 640
	
	#640*368
    center_image = 320
	
    center = (lane_center - center_image)*xm_per_pix#转换成米
    
    if print_data == True:
        #现在的曲率半径已经转化为米了
        print(left_curverad, 'm', right_curverad, 'm', center, 'm')
 
    return left_curverad, right_curverad, center	
	
def show_info(img,left_cur,right_cur,center):
    #在图片中显示出曲率
    cur = (left_cur + right_cur) / 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 使用默认字体
    cv2.putText(img,'Curvature = %d(m)' % cur,(50,50),font,0.5,(255,255,255),1)
    #照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
    #添加文字
    
    if center < 0:
        fangxiang = 'left'
    else:
        fangxiang = 'right'
        
    cv2.putText(img,'the angle is %.2fm of %s'%(np.abs(center),fangxiang),(50,100),font,0.5,(255,255,255),1)		

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
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
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
	
# get video frame
cap = cv2.VideoCapture('video.mp4')
nbframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Num. Frames = ', nbframes)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output-1.avi',fourcc, fps, (w,h))

while(cap.isOpened()):
	if int(cap.get(cv2.CAP_PROP_POS_FRAMES))==1140:
		ret,frame = cap.read()
		cv2.imshow('first frame', frame)
		cv2.imwrite('./output_images/first_frame.jpg',frame)
		
		# binary
		undist = undistort(frame)
		x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
		color_transforms = hls_select(undist,thresh=(150,255))
		
		color_x_sobel = np.zeros_like(x_sobel)
		color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1
		cv2.imshow("binary_output",color_x_sobel*255)
		
		# perspective transformation
		warped_img,unpersp, Minv = warp(color_x_sobel)
		cv2.imshow("pt",warped_img*255)
		cv2.imwrite('./output_images/warped_img1.jpg',warped_img*255)
		
		# 
		find_line_image,left_fit,right_fit = find_lines(warped_img)
		
		left_curverad, right_curverad, center = curvature(left_fit,right_fit,find_line_image)
		
		result = draw_lines(undist,warped_img,left_fit,right_fit,left_curverad,right_curverad,center)
		
		#out.write(result)
		
		if cv2.waitKey(0) & 0xFF == ord('q'):
			#continue
			break
		
	else:
		# 30s and 49s are bad
		
		ret,frame = cap.read()
		# undist = undistort(frame)
		# x_sobel = abs_sobel_thresh(undist,thresh = (22,100))
		# color_transforms = hls_select(undist,thresh=(150,255))
		
		# color_x_sobel = np.zeros_like(x_sobel)
		# color_x_sobel[ (color_transforms == 1) | (x_sobel) == 1 ] = 1
		# warped_img,unpersp, Minv = warp(color_x_sobel)
		# find_line_image,left_fit,right_fit = find_lines(warped_img,False)
		
		# left_curverad, right_curverad, center = curvature(left_fit,right_fit,find_line_image)
		
		# result = draw_lines(undist,warped_img,left_fit,right_fit,left_curverad,right_curverad,center,False)
		
		# out.write(result)
		# print(cap.get(cv2.CAP_PROP_POS_FRAMES))

		
cap.release()
#out.release()
cv2.destroyAllWindows()