# Import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import datetime
import time
import os
import imutils
import numpy as np
import glob
import argparse

def plot_cv2(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def snap_detect():
    print("Warming up..")
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    print("Capturing..")
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    image = image[50:480, 150:400]

    print("Detecting...")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)#11, 3)
    filtered = cv2.medianBlur(gray,7) # 13
    edged = cv2.Canny(filtered, 30, 150)

    # Highlighting detections
    (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("I count {} insects in this image".format(len(cnts)))
    edged_image = image.copy()
    cv2.drawContours(edged_image, cnts, -1, (0, 255, 0), 1);

    ResultName = 'detections/Result_' + str(time.strftime("%d_%m_%Y_%H_%M_%S")) + '.jpg'
    cv2.imwrite(ResultName,edged_image)

    camera.close()
    
snap_detect()