import cv2
import numpy as np
img = cv2.imread(r'/Users/luoliang/Desktop/opencv/cornor.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 二值化
_,thrsh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 开运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
opening  = cv2.morphologyEx(thrsh,cv2.MORPH_OPEN,kernel,iterations=2)
# 膨胀操作 背景
bg = cv2.dilate(opening,kernel,iterations=2)
# 前景
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# 正则化
cv2.normalize(dist_transform,dist_transform,0,1,cv2.NORM_MINMAX)
# 二值化处理
_,fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,cv2.THRESH_BINARY)
fg = np.uint8(fg)
# 未知区域
unknown = cv2.subtract(bg,fg)
# 计算makers
_,makers = cv2.connectedComponents(fg)

makers += 1

makers[unknown == 255] = 0
# 分水岭算法
makers = cv2.watershed(img,makers)
# 抠图函数
# img[makers>1] = [0,255,0]
mask = np.zeros(shape=img.shape[:2],dtype=np.uint8)
mask[makers > 1] = 255
coins = cv2.bitwise_and(img,img,mask=mask)
cv2.imshow('img',coins)
cv2.waitKey(0)
cv2.destroyWindow()
