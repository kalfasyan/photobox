#!/home/pi/.virtualenvs/cv/bin/python3
import datetime
import glob
import io
import multiprocessing
import os
import pathlib
import threading
import time
import warnings
from collections import deque
from datetime import datetime

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate as it
from PIL import Image
from scipy.io.wavfile import write
from scipy.signal import savgol_filter

import picamera
import PySimpleGUIWeb as sg
import RPi.GPIO as GPIO
from common import *
from guizero import *
from picamera import PiCamera
from picamera.array import PiRGBArray
from snap_detect import *
from wingbeat_detect import *

warnings.simplefilter("once", DeprecationWarning)


# -------------------------------------------------
# ------------- LOGGER CONFIG ---------------------

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler('traplogs.log'),
                        logging.StreamHandler()])

logger = logging.getLogger(__name__)

def create_expt():
    if not os.path.isdir(default_exp_path):
        os.mkdir(default_exp_path)
    
    name = "test"#app.question("Experimental folder", "Give a name for the experiment.")
    if name is not None:
        created_experiment = f'{default_exp_path}/{name}'
        make_dirs([created_experiment])
        if not os.path.isdir(default_exp_path):
            os.mkdir(default_exp_path)

        global currdir_full
        currdir_full = f"{default_exp_path}/{name}"

        print(f"Current experiment: {currdir_full.split('/')[-1]}")

def record_wingbeats(window):
    wh = WingbeatSensorHandler(show_detections=True, obj_detect=objdetect, currdir_full=currdir_full)
    wh.initialize()
        
    while True:
        wh.newplot = False
        wh.read()
        if wh.newplot:
            time.sleep(.5)
            logging.debug(f"#################### NEW PLOT shape: {wh.plot.shape}")
            imgbytes = cv2.imencode('.png', wh.plot)[1].tobytes()
            window['wbsensor'].update(data=imgbytes)
        if stopsensor_event.is_set():
            logging.debug("####### STOPPED THREAD #############")
            wh.finalize()
            break
    return

def take_picture():
    print("Warming up..")
    phi = PicamHandler(setting='image', currdir_full=currdir_full)
    phi.capture_and_detect(save=True)
    print("Saved image")
    # disp_img = cv2.cvtColor(phi.edged_image,cv2.COLOR_BGR2RGB)
    disp_img = cv2.resize(phi.edged_image, (640,480))
    imgbytes = cv2.imencode('.png', disp_img)[1].tobytes()
    window['image'].update(data=imgbytes)

create_expt()

# def main():
sg.theme('Black')

# define the window layout
layout = [
            [sg.Text('Live Video', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Image(filename='', key='image')],
            [
            sg.Button('Start video', size=(10, 1), font='Helvetica 14'),
            sg.Button('Stop video', size=(10, 1), font='Any 14'),
            sg.Button('Take picture', size=(10, 1), font='Any 14'),
            sg.Button('Exit', size=(10, 1), font='Helvetica 14'), 
            ],

            [sg.Text('Wingbeat Sensor', size=(40, 1), justification='center', font='Helvetica 20')],
            [sg.Image(filename='', key='wbsensor')],
            [
            sg.Button('Start sensor', size=(10, 1), font='Helvetica 14'),
            sg.Button('Stop sensor', size=(10, 1), font='Any 14'),
            ],                
            ]

# create the window and show it without the plot
window = sg.Window('Demo Application - OpenCV Integration',
                    layout, location=(800, 400))

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap = cv2.VideoCapture(0)
video_rec = False
wingb_rec = False
print("SOMETHING")

while True:
    event, values = window.read(timeout=20)

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break#return

    elif event == 'Start video':
        if video_rec:
            logging.debug("Video is already on.")
        else:
            video_rec = True
            phv = PicamHandler(setting='video', currdir_full=currdir_full)
            time.sleep(1) # warm-up time
            for i, frm in enumerate(phv.camera.capture_continuous(phv.rawCapture, format='bgr', use_video_port=True)):
                event, values = window.read(timeout=20)
                image = frm.array
                image = cv2.resize(image, (640,480))
                imgbytes = cv2.imencode('.png', image)[1].tobytes()
                window['image'].update(data=imgbytes)            
                phv.rawCapture.truncate(0)

                if event in ['Stop video', 'Take picture']:
                    phv.camera.close()
                    video_rec = False
                    img = np.full((480, 640), 255)
                    # this is faster, shorter and needs less includes
                    imgbytes = cv2.imencode('.png', img)[1].tobytes()
                    window['image'].update(data=imgbytes)
                    break
                else:
                    continue

    elif event == 'Stop video':
        video_rec = False
        img = np.full((480, 640), 255)
        # this is faster, shorter and needs less includes
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        window['image'].update(data=imgbytes)

    elif event == 'Take picture':
        thread_pic = threading.Thread(target=take_picture)
        thread_pic.start()

    if video_rec:
        ret, frame = cap.read()
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        # print(np.array(frame).shape)
        window['image'].update(data=imgbytes)

    if event == 'Start sensor':
        if wingb_rec:
            logging.debug("Sensor already on.")
        else:
            wingb_rec = True
            stopsensor_event = multiprocessing.Event()#threading.Event()
            thread_wingb = multiprocessing.Process(target=record_wingbeats, args=(window,))
            thread_wingb.start()
        
    elif event == 'Stop sensor':
        wingb_rec = False
        stopsensor_event.set()
        thread_wingb.terminate()
        thread_wingb.join()


    # if wingb_rec:
    #     wh.newplot = False
    #     wh.read()
    #     if wh.newplot:
    #         plot_img = wh.plot#Image.fromarray(wh.plot)
    #         imgbytes = cv2.imencode('.png', plot_img)[1].tobytes()
    #         window['wbsensor'].update(data=imgbytes)

# main()
