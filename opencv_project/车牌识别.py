import cv2
import numpy as np
img = cv2.imread(r'/Users/luoliang/Desktop/opencv/car_number.webp')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

import pytesseract
car = cv2.CascadeClassifier(r'/Users/luoliang/opt/anaconda3/envs/py36/lib/python3.6/site-packages/cv2/data/haarcascade_russian_plate_number.xml')
cars = car.detectMultiScale(img_gray)
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
    roi = img_gray[y:y + h, x:x + w]
    '''二值化'''
    _,thresd = cv2.threshold(roi,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 形态学
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # 开运算
    roi = cv2.morphologyEx(thresd,cv2.MORPH_OPEN,kernel,iterations=2)
    # 腐蚀
    roi = cv2.erode(roi,kernel,iterations=2)
    # 膨胀
    roi = cv2.dilate(roi,kernel,iterations=1)
    # tesseract操作
    # '/opt/homebrew/Cellar/tesseract/5.3.0_1/share/tessdata'

    # , lang = 'chi_sim + eng', config = '--psm 8 --oem 3'

    word = pytesseract.image_to_string(roi, lang='chi_sim+eng', config='--psm 6 --oem 3')
    print(word)
    cv2.imshow('img',roi)
cv2.waitKey(0)
cv2.destroyWindow()