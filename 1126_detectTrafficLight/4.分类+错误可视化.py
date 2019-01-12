import cv2
import helpers
import random
import test_functions

def estimate_label(rgb_image):    # 输入RGB图像后，获取对应的预测输出
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = create_feature(rgb_image)
    
    return predicted_label   

def get_misclassified_images(test_images):    # 输入全体测试图像
    # Track misclassified images by placing them into a list 将误分类的图像放入一个列表中
    misclassified_images_labels = []
 
    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:
 
        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."
 
        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."
 
        # Compare true and predicted labels 
        if(predicted_label != true_label):    # 如果预测标签和实际标签不符，则将该图像、预测标签、实际标签放入列表中
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels

IMAGE_DIR_TEST = "traffic_light_images/test/"    # 载入测试数据
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)
 
# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)    # 将图像和标签标准化
 
# Shuffle the standardized test data    shuffle：洗牌
random.shuffle(STANDARDIZED_TEST_LIST)     # 将标准化后的测试集随机打乱

# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)    # 分类出错的样本
 
# Accuracy calculations    计算准确率
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total
 
print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

print(len(MISCLASSIFIED))
 
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(MISCLASSIFIED[i][0])
    
tests = test_functions.Tests()
 
if(len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)    #测试是否有红灯被误分类为绿灯
else:
    print("MISCLASSIFIED may not have been populated with images.")
