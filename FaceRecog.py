# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:45:42 2019

@author: Sanket
"""

import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier("C:\\Users\\Sanket\\Anaconda3\\pkgs\\opencv-3.4.4-py37hb76ac4c_1203\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
   
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faceSamples=[]
    Ids=[]
    
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[0])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainer/trainer.yml')

print("\n {0} faces trained. Exiting Program".format(len(np.unique(Ids))))