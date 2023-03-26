import cv2
import numpy as np

img = cv2.imread(r'/Users/luoliang/Desktop/opencv/datika.png')
gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯模糊
bluured = cv2.GaussianBlur(gary,(5,5),0)
# 边缘检测
edged = cv2.Canny(bluured,75,200)
# 检测轮廓
cont,_ = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img_cont = img.copy()
cv2.drawContours(img_cont,cont,-1,(0,0,255),2)
doCont = None
if len(cont) > 0:
    cents = sorted(cont,key=cv2.contourArea,reverse=True)
    for c in cents:
        perimter = cv2.arcLength(c,True)
        # 找出图片的4个角点
        approx = cv2.approxPolyDP(c,0.02*perimter,True)
        if len(approx) == 4:
            doCont = approx
            break
def order_pionts(pts):
    rect = np.zeros((4,2),dtype='float32')
    s = pts.sum(axis=1)
    # 找出4个坐标
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    b = np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(b)]
    rect[3] = pts[np.argmax(b)]
    return rect
def four_pionts_transfrom(image,pts):
    rect = order_pionts(pts)
    (tl,tr,bl,br) = rect
    wA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
    wB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
    max_wdith = max(int(wB),int(wA))

    hA = np.sqrt((tr[0] - br[0])**2 + (tr[1]-br[1])**2)
    hB = np.sqrt((tl[0] - bl[0])**2 + (tl[1]-bl[1])**2)
    max_height = max(int(hA), int(hB))

    dst = np.array([[0, 0], [max_wdith - 1, 0], [max_wdith - 1, max_height-1],[0,max_height-1]],dtype='float32')
    # 转化矩阵
    M = cv2.getPerspectiveTransform(rect,dst)

    warped = cv2.warpPerspective(image,M,(max_wdith,max_height))
    return warped

warped = four_pionts_transfrom(gary, doCont.reshape(4,2))
# 二值化处理
thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow('Warped', thresh)
# 找轮廓
Contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
thresh_copy = thresh.copy()
cv2.drawContours(thresh_copy,Contours,-1,255,3)

# 对轮廓进行筛选 找到特定宽高的圆圈
qustion_cont = []
for c in Contours:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    if ar > 0.78 and ar < 0.83 and w >55 and w <65 :
        qustion_cont.append(c)
# print(qustion_cont)
# 轮廓排序
def sort_contours(cnts,method = 'left-to-right'):
    reverse = False
    i = 0
    # 按x轴坐标进行排序
    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    # 按y轴坐标进行排序
    if method == 'bottom-to-top' or method == 'top-to-bottom':
        i = 1
    # 计算最大外接矩形
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]

    (cnts,bounding_boxes)=zip(*sorted(zip(cnts,bounding_boxes),key=lambda b:b[1][i],reverse=reverse))
    return cnts,bounding_boxes
# 按从上到下的进行排序
qustion_cont = sort_contours(qustion_cont,method='top-to-bottom')[0]

# print(qustion_cont)
# # 正确答案
ANSWER_KRY= {0:1,1:4,2:0,3:3,4:1}
bubbled = None
correct = 0
for (q,i) in enumerate(np.arange(0,25,5)):
    cent = sort_contours(qustion_cont[i:i+5])[0]
    for (j,c) in enumerate(cent):
        # print(j,c)
        mask = np.zeros(thresh.shape,dtype='uint8')
        cv2.drawContours(mask,[c],-1,255,-1)
        # 先做与运算，把涂满了的圈 变成全白
        mask = cv2.bitwise_and(thresh,thresh,mask=mask)
        # 全白的mask 0的个数就会很多
        total = cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)
    k = ANSWER_KRY[q]
    if k == bubbled[1]:
        correct += 1
    cv2.drawContours(warped,[cent[k]],-1,(0,0,255),7)

score = (correct/5)*100
print(f'score:{score:.2f}%')
cv2.putText(warped,f'last_score:{score:.2f}%',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

cv2.imshow('mask',mask)
cv2.imshow('Warped Image', warped)
cv2.waitKey(0)
cv2.destroyWindow()




