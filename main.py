# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 09:46:18 2021

@author: VishalWagaraj
"""


# packages we need 
import cv2
import numpy as np
import urllib.request


#List of objects that can be detected using this model

classNames= []
classFilePath = 'data/coco.names'
with open(classFilePath,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
print('Number of objects that can be Detected: ', len(classNames))
print(classNames)

#Creating dnn Detection Model
#locating the trained model class path

configPath = 'data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'data/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


#Lets' test on an image
#importing image from assets directory

img = cv2.imread('assets/dogCat.png')
print(img.shape)


#This image kinda big, let's Resize it


# resize a bit 
imgHeight, imgWidth, _ = img.shape
scale= 60
imgH = int (imgHeight *scale /100)
imgW = int (imgWidth * scale /100)
img = cv2.resize(img, (imgW, imgH))


#Lets' declear some variables
rectColor =(244,5,20)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontSize = 1
fontColor = (255, 0, 255)
rectColor =(244,5,20)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontSize = 1
fontColor = (255, 0, 255)


#Threshold to detect object

thres = 0.45 # Threshold to detect object
nms_thres = .5 # 0-1 higer value means lower suppress 

classIds, confs, bbox = net.detect(img,confThreshold=thres)
type(confs),type(bbox)

#Formating for non-maxima suppression
#it removes overlap bounding boxes & keep the most confident ones. make sure bounding boxex and confident are List of floats, it shouldn't associate with numpy


bbox = list(bbox)
confs = list(np.array(confs).reshape(1, -1)[0])
confs = list(map(float,confs ))

indics = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres) # remove overlaps
print('Number of bjects we get: ',len(indics))


#Take a look on actual image
for ind in indics:
    i = ind[0] # ind returns list we need index only
    box = bbox[i]
    
    cv2.rectangle(img, box, rectColor, 1) #set the rectangle around object
    
    obj_name = classNames[classIds[i][0]-1]
    obj_confidences = round(confs[i]*100, 1)
    print(obj_name, " conf ", str(obj_confidences))
    
    if obj_confidences>56:
        #show accuracy level
        cv2.putText(img, str(obj_confidences), (box[0]+10, box[1]+10), 
                   font, fontSize, fontColor, 1,cv2.LINE_AA)
    
        #putting name of object 
        cv2.putText(img, obj_name.upper(), (box[0]+70, box[1]+10), 
                   font, fontSize, fontColor,1,cv2.LINE_AA)
    
# Lets show the result 
cv2.imshow('img', img)
cv2.waitKey(0)


#Let's test it on webcam

import cv2
#cam = cv2.VideoCapture(0) # my webcam id
#cam = cv2.VideoCapture("http://192.168.0.114:8080/") # my webcam id
cam = cv2.VideoCapture('http://192.168.0.102:8080/video')
# cam = cv2.VideoCapture('assets/testVideo.mp4') # my webcam id
cam.set(3, 1000) # width of camView
cam.set(4, 800) # height of camView
cam.set(10, 150) # brightness 
cv2.destroyAllWindows()

# we can also make a funtion just for image processing 
while True:
    success, img = cam.read()
    if success:
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        
        #Formating data
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float,confs ))

        indics = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_thres) # remove overlaps
        
        for ind in indics:
            i = ind[0] # ind returns list we need index only
            box = bbox[i]
    
            cv2.rectangle(img, box, rectColor, 1) #set the rectangle around object
    
            #show accuracy level
            cv2.putText(img, str(round(confs[i]*100, 1)), (box[0]+10, box[1]+10), 
                       font, fontSize, fontColor, 1)
            #putting name of object 
            cv2.putText(img, classNames[classIds[i][0]-1], (box[0]+70, box[1]+30), 
                       font, fontSize, fontColor,1)
        
    
        cv2.imshow("Object detection", img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    
cam.release()
cv2.destroyAllWindows()