# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 00:14:03 2019

@author: Sanket
"""

import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("C:\\Users\\Sanket\\Anaconda3\\pkgs\\opencv-3.4.4-py37hb76ac4c_1203\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml");


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id==1):
                Id="ABD"
            elif(Id==2):
                Id="Virat"
            elif(Id==3):
                Id="Leo"
            elif(Id==4):
                Id="Me"
        else:
            Id="Unknown"
        cv2.putText(im, str(Id), (x+5,y-5), font, 1, (0,0,255), 2)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()