import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)

img = cv2.imread('car_plate.jpg')

display(img)

plate_cascade = cv2.CascadeClassifier("haarcascades\haarcascade_russian_plate_number.xml")

def detect_plate(img):
    img_copy = img.copy()
    plate_rects = plate_cascade.detectMultiScale(img_copy,scaleFactor=1.2, minNeighbors=3)
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)
    return img_copy

result = detect_plate(img)

display(result)
