import cv2
import pytesseract
import pandas as pd

img = cv2.imread(r'/Users/luoliang/Desktop/opencv/t.jpg')
word = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--psm 6 --oem 3')
print(word)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyWindow()