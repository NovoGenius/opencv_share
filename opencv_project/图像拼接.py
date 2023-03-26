import cv2
import numpy as np
img1 = cv2.imread(r'/Users/luoliang/Desktop/opencv/map1.png')
img2 = cv2.imread(r'/Users/luoliang/Desktop/opencv/map2.png')
img1 = cv2.resize(img1,(640,480))
img2 = cv2.resize(img2,(640,480))
g1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
g2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1,des1 = sift.detectAndCompute(g1,None)
kp2,des2 = sift.detectAndCompute(g2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
goods = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        goods.append(m)
if len(goods) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in goods]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in goods]).reshape(-1, 1, 2)
    H,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5)
else:
    print('no')
    exit()
h1,w1 = img1.shape[:2]
h2,w2 = img2.shape[:2]
img1_pts = np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(-1,1,2)
img2_pts = np.float32([[0,0],[0,h2-1],[w2-1,h2-1],[w2-1,0]]).reshape(-1,1,2)

img1_transform = cv2.perspectiveTransform(img1_pts,H)

result_pts = np.concatenate((img2_pts,img1_transform),axis=0)
[x_min,y_min] =  np.int32(result_pts.min(axis=0).ravel() -1 )
[x_max,y_max] =  np.int32(result_pts.max(axis=0).ravel() +1 )


move_matrix = np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]])
result_img = cv2.warpPerspective(img1,move_matrix.dot(H) ,dsize=(x_max+(-x_min),y_max + (-y_min)))
result_img[-y_min:-y_min+h2,-x_min:-x_min+w2] = img2
cv2.imshow('img',result_img)
cv2.waitKey(0)
cv2.destroyWindow()