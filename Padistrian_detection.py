# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 10:31:58 2021

@author: theas
"""

import cv2

#loading pre trained body haar cascade classifier

body_classifier = cv2.CascadeClassifier("D:/DataScience/AI and DL/study material/Computer Vision/haarcascade_fullbody.xml")

#loading video and capturing in variable

cap = cv2.VideoCapture("D:/DataScience/Class/assignment working/DL/CV/180301_06_B_CityRoam_01.mp4")

#capturing each frame from video to classify padistrian
while cap.isOpened():
    
    #getting each frame and frame rate
    ret , frame = cap.read()
    frame = cv2.resize(frame, None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #passing this gray frame to classifier to detect padistrian
    
    padistrian = body_classifier.detectMultiScale(gray,None,3)
    
    #pltotting bounding box around detected padistrians
    
    for (x,y,w,h) in padistrian:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.imshow("Pedistrian",frame)
    
    if cv2.waitKey(1)==13: #for enter_key
        break

cap.release()
cv2.destroyAllWindows()
