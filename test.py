import time


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math



cap=cv2.VideoCapture(0)
# time.sleep(5)
detecter=HandDetector(maxHands=1)
classfier=Classifier("Model/keras_model.h5","Model/labels.txt")
imgsize=300
offset=20
counter=0

folder="data/d"
labels=["a","b","c","d"]
while True:
    success,img=cap.read()
    imgOutput=img.copy()
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
            prediction,index=classfier.getPrediction(imgwhite,draw=False)
            print(prediction,index)

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)

            imgresize = cv2.resize(imgCrop, (imgsize,hCal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize - hCal) / 2)
            imgwhite[hgap:hCal + hgap, :] = imgresize
            prediction, index = classfier.getPrediction(imgwhite,draw=False)
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("ImageCrop",imgCrop)

        cv2.imshow("ImageCrop", imgwhite)
    cv2.imshow("Image",imgOutput)
    cv2.waitKey(1)

