import cv2
import os

os.chdir('C:/Users/massw/OneDrive/Bureau/Programmation/Python_R/Computer-Vision-with-Python/DATA')

img = cv2.imread('00-puppy.jpg')

while True:
    cv2.imshow('Puppy', img)
    
    # IF WE HAVE WAITED AT LEAST 1 MILESCN AND WE HAVE PRESSED THE ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

