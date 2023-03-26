import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)
cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.resizeWindow('window',1280,840)
class Button:
    def __init__(self,pos,width,height,value):
        self.pos = pos
        self.width = width
        self.height = height
        self.value = value
    def draw(self,img):
        '''画实心矩形'''
        cv2.rectangle(img, (self.pos[0], self.pos[1]), (self.pos[0] + self.width, self.pos[1] + self.height), (255, 255, 255), -1)
        '''画矩形边框'''
        cv2.rectangle(img, (self.pos[0], self.pos[1]), (self.pos[0] + self.width, self.pos[1] + self.height), (0, 0, 0), 3)
        '''画数字'''
        cv2.putText(img, self.value, (self.pos[0] + 30, self.pos[1] + 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 3, 20)
    def check_clik(self,x,y):
        if self.pos[0]< x <self.pos[0] + self.width and self.pos[1] < y < self.pos[1] + self.height:
            cv2.rectangle(img,(self.pos[0]+3,self.pos[1]+3),(self.pos[0]+self.width -3 ,self.pos[1]+self.height -3)
                          ,(255,255,255),-1)
            cv2.putText(img,self.value,(self.pos[0]+25,self.pos[1]+80),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,0),3 )
            return True
        else:
            return False

button_values = [['7','8','9','*'],
                 ['4','5','6','-'],
                 ['1','2','3','+'],
                 ['0','/','.','=']]
button_list = []
for x in range(4):
    for y in range(4):
        x_pos = x * 100 + 800
        y_pos = y * 100 + 150
        button = Button((x_pos,y_pos),100,100,button_values[y][x])
        button_list.append(button)

# 创建一个检测对象
detector = HandDetector(maxHands=2,detectionCon=0.8)

my_equation = ''
delyer_counter = 0
while True:
    flag,img = cap.read()
    img = cv2.flip(img,1)
    if flag:
        # 检测手
        hands, img = detector.findHands(img, flipType=False)
        for button in button_list:
            button.draw(img)
        '''创建显示窗口'''
        cv2.rectangle(img,(800,50),(800+400,50+100),(255,255,255),-1)
        cv2.rectangle(img, (800, 50), (800 + 400, 50 + 100), (50, 50, 50), 3)
        if hands:
            lm = hands[0]['lmList']
            # 取出食指和大拇指的距离
            length,_,img = detector.findDistance(lm[4][0:2],lm[8][0:2],img)
            '''取出手指的坐标'''
            x,y = lm[8][0:2]

            '''根据距离去判断是否点击'''
            if length < 50 and delyer_counter == 0:
                for i,button in enumerate(button_list):
                    if button.check_clik(x,y):
                        my_value = button_values[i % 4][int(i/4)]
                        if my_value == '=':
                            try:
                                my_equation = str(eval(my_equation))
                            except Exception:
                                my_equation = ''
                        else:
                            my_equation += my_value
                            delyer_counter=1

        if delyer_counter != 0:
            delyer_counter += 1
            if delyer_counter > 10:
                delyer_counter = 0
        cv2.putText(img,my_equation,(810,130),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),10 )
        cv2.imshow('img',img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            my_equation = ''
    else:
        print('open fail')
cap.release()
cv2.destroyWindow()

