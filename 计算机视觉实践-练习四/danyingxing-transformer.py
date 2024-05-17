import cv2
import numpy as np

# 读取两幅图像
img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)

# 使用SIFT算法提取并匹配特征点
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key = lambda x:x.distance)

# 提取匹配的特征点
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算单应性矩阵
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 将img2图像进行透视变换
warped_img2 = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# 拼接图像
result = np.zeros((max(img1.shape[0], warped_img2.shape[0]), img1.shape[1] + warped_img2.shape[1]), dtype=np.uint8)
result[:img1.shape[0], :img1.shape[1]] = img1
result[:warped_img2.shape[0], img1.shape[1]:] = warped_img2

# 显示结果
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

