import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import  ImageFont,ImageDraw,Image
# cv2.namedWindow('window',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('window',640,480)
'''视频读取  参数路径改为0则可以访问摄像头'''
# cap = cv2.VideoCapture(r'/Users/luoliang/Desktop/opencv/1.mp4')
# while True:
#     ret,frame = cap.read()
#     if not ret:
#         break
#     cv2.imshow('window',frame)
#     key = cv2.waitKey(10)
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyWindow()
'''视频录制并且存储'''
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# vm = cv2.VideoWriter(r'/Users/luoliang/Desktop/opencv/output.mp4',fourcc,20,(640,480))
# while cap.isOpened():
#     ret,frame = cap.read()
#     if not ret:
#         print('不能存储视频')
#         break
#     vm.write(frame)
#     cv2.imshow('window',frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
# vm.release()
# cv2.destroyWindow()
'''鼠标回调操作'''
# cv2.namedWindow('mouse',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('mouse',640,360)
# def mouse_callback(event,x,y,flags,userdata):
#     print(event,x,y,flags,userdata)
#     if event == 1 :
#         cv2.destroyWindow()
# cv2.setMouseCallback('mouse',mouse_callback,'123')
# img = np.zeros((360,640,3),np.uint8)
# while True:
#     cv2.imshow('mouse',img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# cv2.destroyWindow()
'''trackbar 改变图片颜色'''
# cv2.namedWindow('trackbar',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('trackbar',640,480)
# def callback(value):
#     pass
# cv2.createTrackbar('R','trackbar',0,255,callback)
# cv2.createTrackbar('G','trackbar',0,255,callback)
# cv2.createTrackbar('B','trackbar',0,255,callback)
#
# img = np.zeros((480,640,3),np.uint8)
# while True:
#     r = cv2.getTrackbarPos('R','trackbar')
#     g = cv2.getTrackbarPos('G', 'trackbar')
#     b = cv2.getTrackbarPos('B', 'trackbar')
#     img[:] = [b,g,r]
#     cv2.imshow('trackbar',img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# cv2.destroyWindow()
'''改变图片的颜色空间'''
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/cat1.jpg')
# def callback(value):
#     pass
# cv2.namedWindow('color',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('color',640,480)
#
# color_spaces = [
#     cv2.COLOR_BGR2RGBA,cv2.COLOR_BGR2BGRA,
#     cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV,
#     cv2.COLOR_BGR2YUV
# ]
# cv2.createTrackbar('trackbar','color',0,4,callback)
# while True:
#     index = cv2.getTrackbarPos('trackbar','color')
#     cvt_img = cv2.cvtColor(img,color_spaces[index])
#     cv2.imshow('color',cvt_img)
#     key = cv2.waitKey(10)
#     if key == ord('q'):
#         break
# cv2.destroyWindow()
'''颜色通道的分割与融合'''
# img = np.zeros((200,200,3))
# b,g,r = cv2.split(img)
#
# b[10:100,10:100] = 255
# g[10:100,10:100] = 255
#
# img2 = cv2.merge((b,g,r))
# cv2.imshow('img2',img2)
# cv2.imshow('img',img)
#
# cv2.waitKey(0)
# cv2.destroyWindow()

'''绘制图形'''
# img = np.zeros((640,1280,3),np.uint8)
# '''画直线'''
# cv2.line(img,(0,10),(200,100),(0,0,255),2,4)
# cv2.line(img,(200,180),(120,200),(0,180,255),1,4)
# '''画矩形(起点，终点，颜色，线的宽度，线的毛刺)'''

# cv2.rectangle(img,(225,492),(150,91),(0,0,255),2,4)
# cv2.rectangle(img,(200,180),(120,200),(0,180,255),1,4)
# '''画圆'''
# cv2.circle(img,(220,220),100,(12,20,230),5,16)
# '''画椭圆(中心点，长款半径，椭圆角度，是否画完整，颜色，线宽，毛刺)'''
# cv2.ellipse(img,(200,120),(100,50),0,0,360,(0,230,22),4,16)
# '''画多边形'''
# pts =np.array([(20,12),(56,420),(100,34),(220,132)])
# cv2.polylines(img,[pts],True,(0,230,49))
# '''画填充多边形'''
# pts =np.array([(20,12),(56,420),(100,34),(220,132)])
# cv2.fillPoly(img,[pts],(0,230,49))
# '''绘制文本(无法直接使用中文，使用pillow的包)'''
# cv2.putText(img,'car',(400,400),cv2.FONT_HERSHEY_SIMPLEX,5,(0,230,20))
#
# cv2.imshow('draw',img)
# cv2.waitKey(0)
# cv2.destroyWindow()


'''图像的算术运算'''

# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/cat1.jpg')
# img2 = cv2.imread(r'/Users/luoliang/Desktop/opencv/cat.webp')
#
# img = cv2.resize(img,(480,480))
# img2 = cv2.resize(img2,(480,480))
'''加法  要求两个图片的 长宽和通道数相同'''
# new_img = cv2.add(img,new_img2)
'''减法'''
# new_img = cv2.subtract(img,new_img2)
'''乘法'''
# new_img = cv2.multiply(img,new_img2)
'''除法'''
# new_img = cv2.divide(img,new_img2)
'''图像融合（图片做了线性运算） new_img = img1 * w1 + img2 * w2 + bias'''
# new_img = cv2.addWeighted(img,0.6,img2,0.4,0)
'''非操作 用255 - 当前数值'''
# new_img = cv2.bitwise_not(img)
'''与操作'''
# new_img = cv2.bitwise_and(img,new_img2)
'''或操作'''
# new_img = cv2.bitwise_or(img,new_img2)
'''异或操作'''
# new_img = cv2.bitwise_xor(img,new_img2)

''' 图像的放大与缩小'''
# new_img = cv2.resize(img2, (460,399))
'''图片的上下左右翻转'''
# new_img = cv2.flip(new_img,flipCode=-1)
'''图片的顺时针旋转'''
# new_img = cv2.rotate(new_img,rotateCode=cv2.ROTATE_180)
'''图片的仿射(平移)'''
# x,y,ch = new_img.shape
# M = np.float32([[1,0,0],[0,1,20]])
# new_img = cv2.warpAffine(new_img,M,dsize=(y,x))
'''自动获取仿射矩阵'''
# M= cv2.getRotationMatrix2D((100,200),195,1)
# new_img = cv2.warpAffine(new_img,M,dsize=(y,x))
'''通过坐标进行变化'''
# src = np.float32([[10,23],[45,87],[129,300]])
# dst = np.float32([[40,230],[245,187],[49,100]])
# M = cv2.getAffineTransform(src,dst)
# new_img = cv2.warpAffine(new_img,M,dsize=(y,x))

'''def mouse_callback(event,x,y,flags,userdata):
    print(event,x,y,flags,userdata)
    if event == 1 :
        cv2.destroyWindow()
cv2.setMouseCallback('mouse',mouse_callback,'123')'''
'''透视变换'''
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/img.jpeg')
# img = cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
# cv2.namedWindow('new',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('new',(640,480))
# def mouse_callback(event,x,y,flags,userdata):
#     print(x,y)
#     if event==1:
#         cv2.destroyWindow()
# cv2.setMouseCallback('new',mouse_callback,'111')
# while True:
#     cv2.imshow('new',img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
#
# src = np.float32([[43,170],[0,318],[607,184],[595,344]])
# dst = np.float32([[0,0],[0,480],[630,0],[630,480]])
# M = cv2.getPerspectiveTransform(src,dst)
# new_img = cv2.warpPerspective(img,M,(630,480))
'''卷积操作'''
# cv2.namedWindow('new',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('new',(640,480))
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/3.jpeg')
# new_img = cv2.resize(img,(440,680))
'''卷积核'''
# kernel = np.ones((5,5),np.float32)/25
# new_img = cv2.filter2D(img,-1,kernel)
'''方盒滤波'''
# new_img = cv2.boxFilter(img,-1,(5,5),normalize=True)
'''均值滤波'''
# new_img = cv2.blur(img,(5,5))
'''高斯滤波  常用于符合高斯分布的图'''
# new_img = cv2.GaussianBlur(img,(3,3),sigmaX=1)
'''中值滤波  常用于椒盐形状噪声的图 变清晰'''
# new_img = cv2.medianBlur(img,1)
'''双边滤波用于美颜'''
# new_img = cv2.bilateralFilter(new_img,30,40,40)

'''算子都是用于做边缘检测'''
'''sobel 算子 主要利用一阶导数 计算图像边缘'''
# dx = cv2.Sobel(new_img,-1,dx=1,dy=0,ksize=3)
# dy = cv2.Sobel(new_img,-1,dx=0,dy=1,ksize=3)
# new_img = cv2.add(dx,dy)
'''scharr 算子 和sobel逻辑一致，但是kernel的值不一样'''
# dx = cv2.Scharr(new_img,-1,dx=1,dy=0)
# dy = cv2.Scharr(new_img,-1,dx=0,dy=1)
# new_img = cv2.add(dx,dy)
'''拉普拉斯算子  求二阶导 然后用内积 计算出拉普拉斯的卷积核'''
# new_img = cv2.Laplacian(new_img,-1,ksize=5)
'''canny边缘检测  先算出导数，再用反三角计算出角度，最后用非极大值抑制NMS'''
# new_img = cv2.Canny(new_img,50,120)


'''形态学 计算出图片的形状特征  主要对象是灰度图'''
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/7.jpeg')
# new_img = cv2.resize(img,(840,680))
# new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
'''手动全局二值化  返回两个值'''
# thresh,new_img = cv2.threshold(new_img,170,255,cv2.THRESH_BINARY)
'''自动全局二值化  返回一个值'''
# new_img = cv2.adaptiveThreshold(new_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=11,C=0)
'''腐蚀操作 目的去除边缘的异常像素点'''
# kernel = np.ones((3,3),np.uint8)
# new_img = cv2.erode(new_img,kernel,iterations=1)
'''自动获取腐蚀操作的kernel'''
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# new_img = cv2.medianBlur(new_img,13)
# new_img = cv2.erode(new_img,kernel,iterations=1)
'''膨胀操作 '''
# cv2.dilate(new_img,kernel,iterations=1)
'''开运算 先腐蚀 再膨胀 去除外部的噪点'''
# new_img = cv2.morphologyEx(new_img,cv2.MORPH_OPEN,kernel,iterations=3)
'''闭运算 先膨胀 再腐蚀 去除内部的噪点'''
# new_img = cv2.morphologyEx(new_img,cv2.MORPH_CLOSE,kernel,iterations=1)
'''形态学梯度 梯度= 原图- 腐蚀'''
# new_img = cv2.morphologyEx(new_img,cv2.MORPH_GRADIENT,kernel,iterations=1)
'''顶帽操作 = 原图 - 开运算  = 图像外部去掉的噪点'''
# new_img = cv2.morphologyEx(new_img,cv2.MORPH_TOPHAT,kernel,iterations=1)
'''顶帽操作 = 原图 - 闭运算  = 图像内部的噪点'''
# new_img = cv2.morphologyEx(new_img,cv2.MORPH_BLACKHAT,kernel,iterations=1)
# cv2.imshow('new',new_img)
# cv2.waitKey(0)
# cv2.destroyWindow()

'''图像轮廓  图形分析和物体检测'''
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/9.jpeg')
# img =  cv2.resize(img,(840,680))
'''灰度处理'''
# new_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''自动2值化处理'''
# new_img = cv2.adaptiveThreshold(new_img,255,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=77,C=0)
'''手动2值化处理'''
# x,new_img = cv2.threshold(new_img,120,255,cv2.THRESH_BINARY)
'''查找轮廓'''
# result,contours,hierarchy = cv2.findContours(new_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
'''画出轮廓  先画出轮廓才能进行多边形逼近和凸包'''
# new_img1 = cv2.drawContours(img,contours,0,(0,0,255),1)
'''计算轮廓的周长和面积'''
# area = cv2.contourArea(contours[1])
# print(area)
# perimeter = cv2.arcLength(contours[1],closed=True)
# print(perimeter)
'''多边形逼近  用多边形替代弧线，不断逼近，小于阈值继续切分'''
# approx = cv2.approxPolyDP(contours[0],5,closed=True)
# new_img2 = cv2.drawContours(img,[approx],-3,(255,0,0),1)
'''凸包 用尽可能少的线段去包含'''
# hull = cv2.convexHull(contours[0])
# new_img2 = cv2.drawContours(img,[hull],0,(0,255,0),2)
'''外接矩形  最小外接矩形'''
# min_ares = cv2.minAreaRect(contours[0])
'''绘制矩形'''
# box = cv2.boxPoints(min_ares)
'''把box的坐标转化成整数型'''
# box = np.int0(box)
# new_img2 = cv2.drawContours(img,[box],0,(0,255,0),2)
'''最大外接矩形'''
# x,y,z,h = cv2.boundingRect(contours[0])
# new_img2 = cv2.rectangle(img,(x,y),(x+z,y+h),(0,0,255),2)
# cv2.imshow('new',new_img2)
# cv2.waitKey(0)
# cv2.destroyWindow()

'''图像金字塔 主要用于图像分割'''
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/9.jpeg')
# img = cv2.resize(img,(840,680))
# gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''高斯金字塔'''
'''向上采样'''
# new_img = cv2.pyrDown(img)
# new_img1 = cv2.pyrDown(new_img)
'''向下采样'''
# new_img = cv2.pyrUp(img)
'''拉普拉斯金字塔 由原图减去高斯金字塔先向下采样 然后 向上采样'''
# dst = cv2.pyrDown(img)
# dst1 = cv2.pyrUp(dst)
'''拉普拉斯金字塔'''
# lp = img - dst1
'''残差 + 高斯金字塔 = 原图'''
# dst2 = cv2.add(dst1,img)
# cv2.imshow('new',dst2)
'''统计直方图 x轴为灰度级的值， y轴为 出现的次数'''
# hist  = cv2.calcHist(img,[2],None,[256],(0,255))
# img_dark = img - 30
# img_bright = img + 30
'''查看直方图'''
# img_dark_hist = cv2.calcHist(img_dark,[0],None,[256],(0,255))
'''直方图均衡化'''
# avg_img_dark = cv2.equalizeHist(img_dark)
'''查看直方图'''
# avg_img_dark_hist = cv2.calcHist(avg_img_dark,[0],None,[256],(0,255))
# cv2.imshow('img',np.hstack((img_dark,avg_img_dark)))
# plt.plot(img_dark_hist)

'''使用掩膜直方图'''
# mask = np.zeros(img.shape,np.uint8)
'''设置直方图的统计区域'''
# mask[230 :580,100:640] = 255
# plt.plot(avg_img_dark_hist)
# plt.hist(hist)
# cv2.imshow('img',cv2.bitwise_and(img,img,mask=mask))
# plt.show()

'''特征检测 边缘检测，角检测，区域检测，脊检测 '''
'''Harris 角点检测 缺点当图片放大了 原来是角点的地方 可能就不是角点了'''
# dst = cv2.cornerHarris(gary,blockSize=2,ksize=5,k=0.0 4)
# img[dst> (0.01 * dst.max())] = [0,0,255]
'''SIFT 关键点检测 解决了 harris的缺点'''
# sift = cv2.xfeatures2d_SIFT.create()
'''关键点'''
# kp = sift.detect(gary)
# cv2.drawKeypoints(gary,kp,img)
'''描述子'''
# kp,des = sift.compute(gary,kp)

'''Tomasi 角点检测是 Harris的改进， 不用去手工去定义R值的阿尔法值的大小，采用最小的R= min（浪马达）'''
# dst = cv2.goodFeaturesToTrack(gary,maxCorners=0,qualityLevel=0.01,minDistance=30)
# dst = np.int0(dst)
'''画出角点'''
# for i in dst:
#     x, y = i.ravel()
#     cv2.circle(img,(x,y),3,(255,0,0),-1)

'''SURF 提升了SIFT的算法速度 高版本不能用'''
'''ORB 特征检测 速度最快 可以达到实时检测  准确率有下降'''
# orb = cv2.ORB_create()
# kp = orb.detect(gary)
'''ORB描述子只有32维向量'''
# kp,des = orb.compute(img,kp)
'''画出关键点'''
# cv2.drawKeypoints(gary,kp,img)
'''特征匹配'''
'''暴力特征匹配'''
# img1 = cv2.imread(r'/Users/luoliang/Desktop/opencv/9.jpeg')
# img2 = cv2.imread(r'/Users/luoliang/Desktop/opencv/10.jpeg')
# img1 = cv2.resize(img1,(840,680))
# img2 = cv2.resize(img2,(840,680))
# gary1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# gary2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
'''创建SIFT的对象'''
# sift = cv2.SIFT_create()
# kp1,des1 = sift.detectAndCompute(gary1,None)
# kp2,des2 = sift.detectAndCompute(gary2 ,None)
'''进行暴力特征匹配'''
# bf = cv2.BFMatcher(cv2.NORM_L1)
# match = bf.match(des1,des2)
# result = cv2.drawMatches(img1,kp1,img2,kp2,match,None)
'''FLANN特征匹配  在大数据上面比暴力 效果更好'''
# index_params = dict(algorithm = 1,tree = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matchs = flann.match(des1,des2)
# result = cv2.drawMatches(img1,kp1,img2,kp2,matchs,None)
# cv2.imshow('img',result)
# cv2.waitKey(0)
# cv2.destroyWindow()

'''图像匹配'''
# img_st = cv2.imread(r'/Users/luoliang/Desktop/opencv/img1.png')
# img_p = cv2.imread(r'/Users/luoliang/Desktop/opencv/img2.png')
#
# w,h = img_p.shape[0],img_p.shape[1]
'''用平方的方法去 min_loc,其他方用max_loc'''
'''一对一的场景'''
# res = cv2.matchTemplate(img_st,img_p,method=cv2.TM_CCOEFF_NORMED)
# min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
# cv2.rectangle(img_st,(max_loc[0],max_loc[1]),(max_loc[0]+w,max_loc[1]+h),(0,0,225),2)
'''一对多的场景'''
# res = cv2.matchTemplate(img_st,img_p,method=cv2.TM_CCOEFF_NORMED)
# thsold = 0.8
# loc = np.argwhere(res >= thsold)
# for pt in loc:
#     bottom_right = (pt[0]+w,pt[1]+h)
#     cv2.rectangle(img_st,pt,bottom_right,(0,0,255),2)
# cv2.imshow('img',img_st)
# cv2.waitKey(0)
# cv2.destroyWindow()


'''图像分割'''