# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 22:27:16 2021

@author: Riad
"""

import cv2
import numpy as np

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture("http://192.168.0.101:4747/mjpegfeed")

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Handles the mirroring of the current frame
    frame = cv2.flip(frame,1)
    #frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Our operations on the frame come here

    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()