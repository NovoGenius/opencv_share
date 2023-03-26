import cv2
import numpy as np
cap = cv2.VideoCapture(r'/Users/luoliang/Desktop/opencv/video.mp4')
# cap = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorKNN()
min_w = 100
min_h = 90
line_high = 600
car = []
carno = 0
offset = 9
'''计算外接矩形的中心点,当中心点与线重合就可以计数'''
def center(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = int(x)+x1
    cy = int(y)+y1
    return cx,cy
while True:
    ret,frame = cap.read()
    if ret == True:
        '''灰度处理'''
        gary = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        '''高斯模糊'''
        blur = cv2.GaussianBlur(gary,(3,3),5)
        mask = mog.apply(blur)
        '''腐蚀操作'''
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erode = cv2.erode(mask, kernel, iterations=1)
        '''膨胀回来'''
        dilate = cv2.dilate(erode,kernel,iterations=2)
        '''闭运算 去除内部的噪点'''
        colse = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
        '''画出车辆检测线'''
        cv2.line(frame,(10,line_high),(1200,line_high),(255,255,0),thickness=2)
        '''找车辆轮廓'''
        contours,h = cv2.findContours(colse,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
        '''画出所有选出来的轮廓'''
        for contour in contours:
            '''最大外接矩形'''
            (x,y,w,h) = cv2.boundingRect(contour)
            '''通过外接矩形，过滤小矩形'''
            is_vild = (h>=min_h)&(w >=min_w)
            if not is_vild:
                '''跳过当前循环，继续下一次符合要求的循环 '''
                continue
            '''画矩形'''
            cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),thickness=2)
            '''给车画中心点'''
            cpoint = center(x,y,w,h)
            car.append(cpoint)
            cv2.circle(frame,(cpoint),5,(0,0,255),-1)
            '''检测汽车是否过线'''
            for (x,y) in car:
                if y > (line_high-offset) and y < ( line_high+offset):
                    carno +=1
                    car.remove((x,y))
        '''车流统计数字实时显示'''
        cv2.putText(frame,'Carnumber count:'+str(carno),(300,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
        cv2.imshow('video',frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyWindow()
