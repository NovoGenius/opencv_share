import cv2
import numpy as np
class App:
    def __init__(self,image):
        self.image = image
        self.start_x = 0
        self.start_y = 0
        self.rect_flags = False
        self.rect_flags1 = False
        self.img = cv2.imread(self.image)
        self.img2 = self.img.copy()
        self.rect = (0, 0, 0, 0)
        self.rect1 = (0, 0, 0, 0)
        self.mask = np.zeros(self.img2.shape[:2],dtype=np.uint8)
        self.output = np.zeros(self.img2.shape[:2],dtype=np.uint8)
        self.start_x1 = 0
        self.start_y1 = 0

    def on_mouse(self,event,x,y,flags,param):
        # 按下左键开始框选前景区域
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x = x
            self.start_y = y
            # 按下表示需要进行矩形的画取
            self.rect_flags = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_flags = False
            cv2.rectangle(self.img, (self.start_x, self.start_y), (x, y), (0, 0, 255), 4)
            self.rect = (min(self.start_x,x),min(self.start_y,y),abs(self.start_x-x),abs(self.start_y-y))

        # 鼠标右键
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.start_x1 = x
            self.start_y1 = y
            self.rect_flags1 = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.rect_flags1 = False
            cv2.rectangle(self.img, (self.start_x1, self.start_y1), (x, y), (0, 255, 255), 4)
            self.rect1 = (min(self.start_x1, x), min(self.start_y1, y), abs(self.start_x1 - x), abs(self.start_y1 - y))

        elif event == cv2.EVENT_MOUSEMOVE and self.rect_flags and self.rect_flags1:
            self.img = self.img2.copy()
            # 鼠标左键画的圈
            cv2.rectangle(self.img,(self.start_x,self.start_y),(x,y),(0,255,0),4)
            # 鼠标右键画的圈
            cv2.rectangle(self.img, (self.start_x1, self.start_y1), (x, y), (0, 255, 255), 4)

    def run(self):

        cv2.namedWindow('img')
        # 绑定鼠标事件
        cv2.setMouseCallback('img',self.on_mouse)
        while True:
            cv2.imshow('img',self.img)
            cv2.imshow('output',self.output)
            # 进行切图
            key = cv2.waitKey(1)
            if key == 27:
                break
                # 按G键开始进行抠图
            elif key == ord('g'):
                cv2.grabCut(self.img2,self.mask,self.rect,None,None,2,cv2.GC_INIT_WITH_RECT)
                # 把可能是前景和前景的切出来
                mask2 = np.where((self.mask == 1) | (self.mask == 3),255,0).astype(np.uint8)
                self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask2)
                # 按V键开始进行抠图
            elif key == ord('v'):
                # 把右键的区域赋予mask=2，为不确定是否为背景，在重新进行计算
                self.mask[self.rect1[1]:self.rect1[1] + self.rect1[3],self.rect1[0]:self.rect1[0]+self.rect1[2]] = 2
                cv2.grabCut(self.img2,self.mask,None,None,None,2,cv2.GC_INIT_WITH_MASK)
                mask3 = np.where((self.mask == 1) | (self.mask == 3), 255, 0).astype(np.uint8)
                cv2.rectangle(self.img,(self.rect1[0],self.rect1[1]),
                              (self.rect1[0]+self.rect1[2],self.rect1[1]+self.rect1[3])
                              ,(255,255,0),3)
                self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask3)
                # 按y执行代表这个区域是前景
            elif key == ord('y'):
                # 把右键的区域赋予mask=2，为不确定是否为背景，在重新进行计算
                self.mask[self.rect1[1]:self.rect1[1] + self.rect1[3],self.rect1[0]:self.rect1[0]+self.rect1[2]] = 1
                cv2.grabCut(self.img2,self.mask,None,None,None,2,cv2.GC_INIT_WITH_MASK)
                mask3 = np.where((self.mask == 1) | (self.mask == 3), 255, 0).astype(np.uint8)
                cv2.rectangle(self.img,(self.rect1[0],self.rect1[1]),
                              (self.rect1[0]+self.rect1[2],self.rect1[1]+self.rect1[3])
                              ,(255,255,0),3)
                self.output = cv2.bitwise_and(self.img2, self.img2, mask=mask3)



        cv2.destroyWindow()

app = App(r'/Users/luoliang/Desktop/opencv/test1.jpeg')
app.run()
