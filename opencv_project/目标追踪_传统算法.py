import cv2
import numpy as np

cap = cv2.VideoCapture(r'/Users/luoliang/Desktop/opencv/video.mp4')

OPENCV_OBJECT_TRACKERS = {
    'boosting':cv2.legacy.TrackerBoosting_create,
    'kcf':cv2.legacy.TrackerKCF_create,
    'mil':cv2.legacy.TrackerMIL_create,
    'csrt':cv2.legacy.TrackerCSRT_create,
    'tld':cv2.legacy.TrackerTLD_create,
    'mosse':cv2.legacy.TrackerMOSSE_create,
    'medianflow':cv2.legacy.TrackerMedianFlow_create

}
trackers = cv2.legacy.MultiTracker_create()
while True:
    rec,frame = cap.read()
    if frame is None:
        break
    success,boxes = trackers.update(frame)
    for box in boxes:
        (x,y,w,h) = [int(v) for v in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('frame',frame)
    key = cv2.waitKey(100)
    if key == ord('s'):
        roi = cv2.selectROI('frame',frame,showCrosshair=False,fromCenter=False)
        # 创建一个追踪器
        tracker = OPENCV_OBJECT_TRACKERS['kcf']()
        trackers.add(tracker,frame,roi)


    if key == ord('q'):
        break
cap.release()
cv2.destroyWindow()

