import numpy as np
import cv2 
import matplotlib.pyplot as plt
%matplotlib inline

nadia = cv2.imread('Nadia_Murad.jpg',0)
denis = cv2.imread('Denis_Mukwege.jpg',0)
solvay = cv2.imread('solvay_conference.jpg',0)

plt.imshow(nadia,cmap='gray')
plt.imshow(denis,cmap='gray')
plt.imshow(solvay,cmap='gray')

# =============================================================================
# FACE DETECTION
# =============================================================================
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def detect_face(img):
    
  
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

result = detect_face(denis)
plt.imshow(result,cmap='gray')

result = detect_face(nadia)
plt.imshow(result,cmap='gray')

# Gets errors!
result = detect_face(solvay)
plt.imshow(result,cmap='gray')

def adj_detect_face(img):
    
    face_img = img.copy()
  
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5) 
    
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img

# Doesn't detect the side face.
result = adj_detect_face(solvay)
plt.imshow(result,cmap='gray')

# =============================================================================
# Eye Cascade File
# =============================================================================
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def detect_eyes(img):
    
    face_img = img.copy()
  
    eyes = eye_cascade.detectMultiScale(face_img) 
    
    
    for (x,y,w,h) in eyes: 
        cv2.rectangle(face_img, (x,y), (x+w,y+h), (255,255,255), 10) 
        
    return face_img
    
result = detect_eyes(nadia)
plt.imshow(result,cmap='gray')

eyes = eye_cascade.detectMultiScale(denis)

# White around the pupils is not distinct enough to detect Denis' eyes here!
result = detect_eyes(denis)
plt.imshow(result,cmap='gray')

cap = cv2.VideoCapture(0) 

while True: 
    
    ret, frame = cap.read(0) 
     
    frame = adj_detect_face(frame)
 
    cv2.imshow('Video Face Detection', frame) 
 
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
        
cap.release() 
cv2.destroyAllWindows()














