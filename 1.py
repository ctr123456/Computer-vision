import cv2
import numpy as np

def stitch_images(image_path1, image_path2, output_path):
    """
    使用特征匹配和单应性矩阵将两张图像拼接在一起。
    Args:
    image_path1 (str): 第一张图像的路径。
    image_path2 (str): 第二张图像的路径。
    output_path (str): 拼接后图像的保存路径。
    """
    # 读取图像
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    # 将图像转换为灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # 创建SIFT检测器
    sift = cv2.SIFT_create()
    # 使用SIFT检测特征点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    # 使用暴力匹配器匹配特征点，使用L2范数
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.match(descriptors1, descriptors2)
    # 根据距离排序匹配点
    matches = sorted(matches, key=lambda x: x.distance)
    # 选择最佳匹配点
    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]
    # 提取匹配点的位置
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # 计算单应性矩阵并变换图像
    homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
    warped_image2 = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    # 合并图像
    result = np.zeros((warped_image2.shape[0], warped_image2.shape[1], 3), dtype=np.uint8)
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    result[0:warped_image2.shape[0], img1.shape[1]:] = warped_image2[0:warped_image2.shape[0], img1.shape[1]:]
    # 保存拼接后的图像
    cv2.imwrite(output_path, result)
if __name__ == '__main__':
    stitch_images('1.png', '2.png', 'output_image.jpg')
