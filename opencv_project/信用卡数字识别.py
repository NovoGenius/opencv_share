import cv2
import numpy as np
img_ref = cv2.imread(r'/Users/luoliang/Desktop/opencv/card_d.png')
img_recon = cv2.imread(r'/Users/luoliang/Desktop/opencv/card_img.png')
'''灰度化处理'''
g1 = cv2.cvtColor(img_ref,cv2.COLOR_BGR2GRAY)

'''二值化处理'''
_,img_ref_binary = cv2.threshold(g1,255,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
'''查找轮廓'''
ref_contours,_= cv2.findContours(img_ref_binary.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_ref,ref_contours,-1,(255,0,0),3)
'''找出最大外接矩形的x坐标 来进行排序'''
'''在轮廓中 计算出每个轮廓的最大矩形  里面是无序的 不是按照0-9的顺序'''
bounding_boxs = [cv2.boundingRect(c) for c in ref_contours]
'''所以需要根据sort进行 大小的排序 依次找出0-9的轮廓参数，参考最大矩形的x点与（0，0）的x轴大小辅助判断'''
(ref_contours,bounding_boxs) = zip(*sorted(zip(ref_contours,bounding_boxs),key=lambda b:b[1][0]))
digits = {}
'''再按照0-9的顺序计算出每个数字的轮廓边界参数'''
for (i,c) in enumerate(ref_contours):
    (x,y,w,h) = cv2.boundingRect(c)
    roi = g1[y:y+h,x:x+w]
    roi = cv2.resize(roi,(57,88))
    '''将黑白颜色进行反转'''
    roi = cv2.bitwise_not(roi)
    digits[i] = roi
    # cv2.imshow('222',digits[0])


'''对所需要的识别的图片进行求解'''
# 按照比例对原图进行resise
h,w = img_recon.shape[0],img_recon.shape[1]
width = 300
r = width/w
img = cv2.resize(img_recon,(300,int(h*r)))
g2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''顶帽操作 突出更明亮的区域'''
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
tophat = cv2.morphologyEx(g2,cv2.MORPH_TOPHAT,rect_kernel)
'''Sobel 进行边缘检测'''
grad_x = cv2.Sobel(tophat,ddepth=cv2.CV_32F,dx=1,dy=0)
grad_x = np.absolute(grad_x)

'''把grad_x 变成0-255 之间的整数'''
min_val,max_val = np.min(grad_x),np.max(grad_x)
grad_x = ((grad_x-min_val)/(max_val-min_val))*255
grad_x = grad_x.astype('uint8')
'''闭操作 先膨胀再腐蚀  希望数字部分能够在一起'''
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,4))
grad_x = cv2.morphologyEx(grad_x,cv2.MORPH_CLOSE,close_kernel)
'''通过OTSU 进行全局二值化'''
_,thresh = cv2.threshold(grad_x,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
'''再来一个闭操作'''
thrsh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,2))
thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,thrsh_kernel)

'''查找轮廓'''
thresh_contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img.copy(),thresh_contours,-1,(0,0,255),3)
'''遍历轮廓找到 过滤出数字的4个轮廓'''
locs = []
for c in thresh_contours:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    # print(ar)
    # print(w,h)
    if ar > 3 and ar < 4:
        if (w < 50 and w >42) and (h < 16 and h > 10):
            locs.append((x,y,w,h))

locs = sorted(locs,key=lambda x:x[0])
'''[(31, 106, 48, 14), (95, 106, 48, 14), (159, 106, 48, 14), (223, 106, 49, 14)]'''

for (gx,gy,gw,gh) in locs:
    group = g2[gy-5:gy+gh+5,gx-5:gx+gw+5]
    # cv2.imshow('dd', group)
    group = cv2.threshold(group,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    img_contours,_ = cv2.findContours(group,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img2_bounding_boxes =[cv2.boundingRect(c) for c in img_contours]
    (img_contours,_) = zip(*sorted(zip(img_contours,img2_bounding_boxes),key=lambda b:b[1][0]))
    group_output = []
    for c in img_contours:
        # print(c)
        (x,y,w,h) = cv2.boundingRect(c)
        roi = group[y:y+h,x:x+w]
        roi = cv2.resize(roi,(57,88))
        # cv2.imshow('as',roi)
        scores = []
        for (digit,digit_roi) in digits.items():
            result = cv2.matchTemplate(roi, digit_roi,cv2.TM_SQDIFF_NORMED)
            (_,score,_,_ )= cv2.minMaxLoc(result)
            scores.append(score)
        group_output.append(str(np.argmin(scores)))
    cv2.rectangle(img,(gx-5,gy-10),(gx+gw+5,gy+gh+5),(0,0,255),1)
    cv2.putText(img,''.join(group_output),(gx,gy-15),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255))

cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyWindow()