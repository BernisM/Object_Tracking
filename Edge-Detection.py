import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

img = cv2.imread('sammy_face.jpg')

edges = cv2.Canny(img, 127, 127)
plt.imshow(edges)

edges = cv2.Canny(img, 0, 255)
plt.imshow(edges)

# Calculate the median pixel value
med_val = np.median(img)
# Lower bound is either 0 or 70% of the median value, whicever is higher
lower = int(max(0,0.7*med_val))
# Upper bound is either 255 or 30% above the median value, whichever is lower
upper = int(min(255,1.3*med_val))

edges = cv2.Canny(img, lower, upper+100)
plt.imshow(edges)

blurred_img = cv2.blur(img, (5,5))
edges = cv2.Canny(blurred_img, lower, upper+100)
plt.imshow(edges)
