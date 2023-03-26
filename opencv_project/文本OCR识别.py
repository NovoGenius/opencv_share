import cv2
import numpy as np
import pytesseract
from PIL import Image
img = cv2.imread(r'/Users/luoliang/Desktop/opencv/收据1.png')
ratio = img.shape[0]/500

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyWindow()
def resize(img,width=None,height=None,inter = cv2.INTER_AREA):
    dim = None
    (h,w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        ar = height/float(h)
        dim = (int(w*ar),height)
    elif height is None:
        ar = width/float(w)
        dim = (width,int(ar*h))
    resized = cv2.resize(img,dim,interpolation=inter)
    return resized

img_resize = resize(img,width=None,height=500)

gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(3,3),0)

edged = cv2.Canny(gray,75,200)

cnts = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
img_cents = cv2.drawContours(gray,cnts,-1,(0,0,255),3)

max_areas = None
for c in cnts:
    per = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*per,True)
    areas = cv2.contourArea(approx)
    # 使用冒泡排序把最大面积的矩形和对应的坐标拿出来
    if len(approx) == 4:
        if max_areas is None or areas > max_areas[0]:
            max_areas = (areas,approx)

img_cents = cv2.drawContours(img_resize,[max_areas[1]],-1,(0,0,255),3)
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

warped = four_pionts_transfrom(img_resize,max_areas[1].reshape(4,2))

warped_gary = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)

ref = cv2.threshold(warped_gary,100,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

cv2.imwrite(r'/Users/luoliang/Desktop/opencv/收据_检测.jpg',ref)
img = Image.open(r'/Users/luoliang/Desktop/opencv/收据_检测.jpg')
word = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6 --oem 3')
print(word)
cv_show('img',ref)
