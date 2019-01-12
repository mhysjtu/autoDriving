import helpers 
import matplotlib.pyplot as plt

IMAGE_DIR_TRAINING = "traffic_light_images/training/"    # 训练数据
IMAGE_DIR_TEST = "traffic_light_images/test/"    # 测试数据
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)    # 导入的训练数据

# IMAGE_LIST 列表中的每一个元素是一个二元组tuple，二元组中第一个元是图像的RGB灰度，第二个元是对应的label（红灯、黄灯、绿灯）
# 黄灯的一个例子
yellow_image = IMAGE_LIST[750][0]
print("yellow_image.shape = ",yellow_image.shape)    # 查看图片大小
plt.subplot(121)
plt.title("yellow image")
plt.imshow(yellow_image)
 # 红灯的一个例子
red_image = IMAGE_LIST[0][0]
print("red_image.shape = ",IMAGE_LIST[0][0].shape)
plt.subplot(122)
plt.title("red image")
plt.imshow(red_image)
plt.show()