import time


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math



cap=cv2.VideoCapture(0)
# time.sleep(5)
detecter=HandDetector(maxHands=1)
imgsize=300
offset=20
counter=0

folder="data/d"
while True:
    success,img=cap.read()
    hands,img=detecter.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255

        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgcropshape=imgCrop.shape



        aspectratio=h/w
        if aspectratio>1:
            k=imgsize/h
            wCal=math.ceil(k*w)

            imgresize=cv2.resize(imgCrop,(wCal,imgsize))
            imgresizeshape = imgresize.shape
            wgap=math.ceil((imgsize-wCal)/2)
            imgwhite[:, wgap:wCal+wgap] =imgresize
        else:
            k = imgsize / w
            hCal = math.ceil(k * h)

            imgresize = cv2.resize(imgCrop, (imgsize,hCal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hgap:hCal + hgap, :] = imgresize


        cv2.imshow("ImageCrop",imgCrop)

        cv2.imshow("ImageCrop", imgwhite)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    if key ==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)
