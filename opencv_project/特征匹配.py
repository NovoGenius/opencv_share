import cv2
import numpy as np
img1 = cv2.imread(r'/Users/luoliang/Desktop/opencv/img2.png')
img2 = cv2.imread(r'/Users/luoliang/Desktop/opencv/img1.png')
# 灰度化
g1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# 计算特征值
sift = cv2.SIFT_create()
kp1 , des1 = sift.detectAndCompute(g1,None)
kp2 , des2 = sift.detectAndCompute(g2,None)
# 创建特征匹配器
index_params = dict(algorithm = 1,tree = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

# 进行特征匹配
matches = flann.knnMatch(des1,des2,k=2)
print(matches)
goods = []
# 过滤特征
for (m,n) in matches:
    if m.distance < 0.75 * n.distance:
        goods.append(m)
if len(goods) >= 4:
    src_points = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
    # 计算H单应性矩阵
    H , _ = cv2.findHomography(src_points,dst_points,cv2.RANSAC,5)
    # 原图坐标
    h,w = img1.shape[:2]
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    # 将小图的坐标经过H变化到大图中
    dst = cv2.perspectiveTransform(pts,H)
    cv2.polylines(img2,[np.int32(dst)],True,(0,0,255),2)

else:
    print('not enough position to compute')
    exit()
ret = cv2.drawMatchesKnn(img1,kp1,img2,kp2,[goods],None)
cv2.imshow('img',ret)
cv2.waitKey(0)
cv2.destroyWindow()