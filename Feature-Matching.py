import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

def display(img,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')

reeses = cv2.imread('reeses_puffs.png',0)
display(reeses)

reeses.shape

cereals = cv2.imread("many_cereals.jpg",0)
display(cereals)

# =============================================================================
# Brute Force Detection with ORB Descriptors
# =============================================================================

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(reeses,None)
kp2, des2 = orb.detectAndCompute(cereals,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 25 matches.
reeses_matches = cv2.drawMatches(reeses,kp1,cereals,kp2,matches[:25],None,flags=2)

display(reeses_matches)

# =============================================================================
# Brute-Force Matching with SIFT Descriptors and Ratio Test
# =============================================================================

# Create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(reeses,None)
kp2, des2 = sift.detectAndCompute(cereals,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)


# Apply ratio test
good = []

# LESS DISTANCE  == BETTER MATCH
# RATIO MATCH 1 < 75% MATCH 2
for match1,match2 in matches:
    #IF MATCH 1 DISTANCE IS LESS THAN 75% OF MATCH 2 DISTANCE
    # THEN DESCRIPTION WAS A GOOD MATCH, LET'S KEEP IT
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

# cv2.drawMatchesKnn expects list of lists as matches.
sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)

display(sift_matches)
