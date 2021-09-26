# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 10:16:04 2021

@author: theas
"""
#---In openCV positive x axis is same but y axis is altered
#origing(0,0) starts from upper left corner 

#////////////////////////CHAPTER 1//////////////////////////////////
import cv2

#--------reading image

img= cv2.imread('D:/DataScience/AI and DL/study material/Computer Vision/mandrill.tif')
cv2.imshow("output", img)
cv2.waitKey(0)

#------importing video
import cv2
cap= cv2.VideoCapture("D:/DataScience/AI and DL/study material/Computer Vision/Day 5/Opencv_vehicle_detection.mp4")

#video is nothing but sequence of many images, to work on it we need to get each frame
#we can do it by while loop
while True:
    success , img =cap.read() #success is boolean true or false returned by cap,img are frames
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


#----reading webcam
cap= cv2.VideoCapture(0) #0 is for web cam

cap.set(3, 640) #width id no 3
cap.set(4,480) #height id no 4
cap.set(10,100) #brightness id is 10
while True:
    success , img =cap.read() #success is boolean true or false returned by cap,img are frames
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

###///////////////////////////////chpter 2/////////////////////

import cv2

img= cv2.imread("D:/DataScience/AI and DL/study material/Computer Vision/mandrill.tif")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",imgGray)
cv2.waitKey(0)

### blurr
import cv2

img= cv2.imread("D:/DataScience/AI and DL/study material/Computer Vision/mandrill.tif")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlurr = cv2.GaussianBlur(imgGray, (7,7) ,0)
cv2.imshow("gray",imgGray)
cv2.imshow("Blurr",imgBlurr)
cv2.waitKey(0)

### edge detector
import cv2
import numpy as np
kernel = np.ones((5,5),unit8)

img= cv2.imread("D:/DataScience/AI and DL/study material/Computer Vision/mandrill.tif")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlurr = cv2.GaussianBlur(imgGray, (7,7) ,0)
imgCanny = cv2.Canny(img,200,300) #increase threshold values for less no of edges
imgDialation = cv2.dilate(imgCanny,kernel,itrations =1)
imgEroded = cv2.erode(imgDilation,kernel,iterations = 1)
cv2.imshow("gray",imgGray)
cv2.imshow("Blurr",imgBlurr)
cv2.imshow("canny",imgCanny)
cv2.imshow("dilarion",imgDialation) #making edges with bigger
cv2.imshow("erodw",imgEroded) #making edges small
cv2.waitKey(0)


#///////////////////////chapter-3////////////////////
#resizing and cropping
import cv2
img= cv2.imread("D:/DataScience/AI and DL/study material/Computer Vision/mandrill.tif")
img.shape
imgResize = cv2.resize(img,(200,200)) # W*H
imgCrop = img[0:200,100:300]   #H*W
cv2.imshow("Original",img)
cv2.imshow("resize",imgResize)
cv2.imshow("Cropped",imgCrop)
cv2.waitKey(0)


#\\\\\\\\\\\\\\\\\\\chapter-4//////////////////
#shape and Text
import cv2
import numpy as np
img=np.zeros((512,512,3))
#print(img.shape)
#img[:]=255,0,0  #blue image

#cv2.line(img,(0,0),(300,300),(0,255,0),3) #drawing shapes on image
cv2.line(img,(0,0),(img.shape[0],img.shape[1]),(0,255,0),3)
#cv2.rectangle(img,(0,0),(350,350),(0,0,225),2) #start point and end diagonal point
cv2.rectangle(img,(0,0),(350,350),(0,0,225),cv2.FILLED)
#cv2.circle(img,(400,50),30,(255,255,0),2)
cv2.circle(img,(400,50),30,(255,255,0),cv2.FILLED)
cv2.putText(img,"ASHPAK",(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)


cv2.imshow("img",img)
cv2.waitKey(0)

#////////////////////////Chapter-5 //////////////
# wrap Prespective

import cv2
import numpy as np

img=cv2.imread("D:/DataScience/AI and DL/study material/Computer Vision/cards.jpg")

width,height = 250,350
pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutputs=cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("orig",img)
cv2.imshow("Output",imgOutputs)

cv2.waitKey(0)

#///////////////////////chapter-6/////////////////
#staking images
import cv2
import numpy as np


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread('Resources/lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))

# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)
cv2.imshow("ImageStack",imgStack)

cv2.waitKey(0)
#/////////////////////////chapter-7///////////////
#trackbar
import cv2
import numpy as np

def empty(a):
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



path = 'Resources/lambo.png'
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
cv2.createTrackbar("Val Min","TrackBars",153,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)


    # cv2.imshow("Original",img)
    # cv2.imshow("HSV",imgHSV)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Result", imgResult)

    imgStack = stackImages(0.6,([img,imgHSV],[mask,imgResult]))
    cv2.imshow("Stacked Images", imgStack)

    cv2.waitKey(1)

#/////////////////////////chapter-8///////////////
#EDGE DETECTION
import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"



            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)




path = 'Resources/shapes.png'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny = cv2.Canny(imgBlur,50,50)
getContours(imgCanny)

imgBlank = np.zeros_like(img)
imgStack = stackImages(0.8,([img,imgGray,imgBlur],
                            [imgCanny,imgContour,imgBlank]))

cv2.imshow("Stack", imgStack)

cv2.waitKey(0)


####################### Chapter-9 ################
import cv2

faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
img = cv2.imread('Resources/lena.png')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("Result", img)
cv2.waitKey(0)

















