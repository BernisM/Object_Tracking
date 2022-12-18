import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

img = cv2.imread('internal_external.png',0)
plt.imshow(img,cmap="gray")