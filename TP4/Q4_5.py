# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:57:45 2022

@author: mjz61
"""

import numpy as np
import cv2
from collections import defaultdict

roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True

cap = cv2.VideoCapture('../Antoine_Mug.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

cpt = 1



gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
gX_roi = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0)
gY_roi = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1)
magnitude_roi = np.sqrt((gX_roi ** 2) + (gY_roi ** 2))
magnitude_roi/=magnitude_roi.max()
orientation_roi = np.arctan2(gY_roi, gX_roi) * 180 // np.pi
lower_mag=np.array([.1])
higher_mag=np.array([1.])
mask_roi = cv2.inRange(magnitude_roi, lower_mag, higher_mag)
img_center = [int(roi.shape[0] / 2), int(roi.shape[1] / 2)]
r_table = defaultdict(list)
for (i, j), value in np.ndenumerate(mask_roi): 
    if value:
        r_table[orientation_roi[i, j]].append((img_center[0] - i, img_center[1] - j))



while(1):
    ret ,frame = cap.read()
    if ret == True:
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	# Backproject the model histogram roi_hist onto the 
	# current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        #dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to dst to get the new location
        
        gray_tracked = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        gX_tracked = cv2.Sobel(gray_tracked, cv2.CV_64F, 1, 0)
        gY_tracked = cv2.Sobel(gray_tracked, cv2.CV_64F, 0, 1)
        orientation_tracked = np.arctan2(gY_tracked, gX_tracked) * 180 // np.pi
        #orientation_tracked = np.arctan2(gY_tracked, gX_tracked) // (2*np.pi)
        magnitude_tracked = np.sqrt((gX_tracked ** 2) + (gY_tracked ** 2))
        magnitude_tracked /= magnitude_tracked.max()
        mask_tracked = cv2.inRange(magnitude_tracked, lower_mag, higher_mag)
        
        #update R-Table
        '''
        roi=magnitude_tracked[c:c+w,r:r+h]
        mask_roi_tracked = cv2.inRange(roi, lower_mag, higher_mag)
        img_center = [int(roi.shape[0] / 2), int(roi.shape[1] / 2)]
        r_table = defaultdict(list)
        for (i, j), value in np.ndenumerate(mask_roi_tracked): 
            if value:
                r_table[orientation_tracked[i, j]].append((img_center[0] - i, img_center[1] - j))
        '''
        hough = np.zeros((frame.shape[0]+int(0.2*frame.shape[0]), frame.shape[1]+int(0.2*frame.shape[0])))
        for (i, j), value in np.ndenumerate(mask_tracked): 
            if value:
                vectors=r_table[orientation_tracked[i,j]]
                for vector in vectors:
                    hough[vector[0]+i, vector[1]+j] += 1
        hough /= hough.max()
        #Arg max            
        #r_index, c_index = np.unravel_index(hough.argmax(), hough.shape)
        #track_window=r_index-h//2,c_index-w//2,h,w
        
        #Mean-shift
        ret, track_window = cv2.meanShift(hough, track_window, term_crit)
        
        
        r,c,h,w=track_window
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)
        cv2.imshow('Hough',hough)
        cv2.imshow('magnitude_tracked',magnitude_tracked)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()