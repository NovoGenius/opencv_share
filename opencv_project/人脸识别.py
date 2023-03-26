import cv2
import numpy as np
# img = cv2.imread(r'/Users/luoliang/Desktop/opencv/test1.jpeg')
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture(0)
cv2.namedWindow('img')
while True:
    rec,img = cap.read()
    img = cv2.flip(img, 1)
    if not rec:
        break

    '''级联器 '''
    eye = cv2.CascadeClassifier(
        r'/Users/luoliang/opt/anaconda3/envs/py36/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
    facer = cv2.CascadeClassifier(
        r'/Users/luoliang/opt/anaconda3/envs/py36/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

    # 检测人脸
    face = facer.detectMultiScale(img)
    # 画出人脸框
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # 切出人脸区域
        roi_face = img[y:y + h, x:x + w]
        # 检测眼睛
        eyes = eye.detectMultiScale(roi_face)
        for (gx, gy, gw, gh) in eyes:
            cv2.rectangle(roi_face, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 3)
            img_eye = roi_face[gy:gy + gh, gx:gx + gw]
            img[y:y + h, x:x + w] = roi_face
    cv2.imshow('img', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyWindow()



# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyWindow()