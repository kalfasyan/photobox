#!/home/pi/.virtualenvs/cv/bin/python3
import glob
import io
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
import RPi.GPIO as GPIO
from common import *
from guizero import *
from picamera import PiCamera
from picamera.array import PiRGBArray
from snap_detect import *

currdir_full = f'{default_ses_path}/{datetime.now().strftime("%Y%m%d")}'
if not os.path.isdir(default_ses_path):
    os.mkdir(default_ses_path)
if not os.path.isdir(currdir_full):
    os.mkdir(currdir_full)

warnings.simplefilter("once", DeprecationWarning)

# -------------------------------------------------
# ------------- LOGGER CONFIG ---------------------

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        logging.FileHandler('traplogs.log'),
                        logging.StreamHandler()])

logger = logging.getLogger(__name__)

#---------------------------------------------------
# ------------- START CAMERA FUNCTIONS -------------

def snap_detect():
    if stop_video_button.enabled:
        if yesno("Video Interruption","You can interrupt video and then take a picture.\nInterrupt video?"):
            stop_video()
        # else:
        #     return None
    if len(platedate_str.value):
        print("Warming up..")
        print(platedate_str.value)
        phi = PicamHandler(setting='image', currdir_full=currdir_full, plateloc=plateloc_bt.value ,platedate=platedate_str.value)
        # phi.capture_and_detect(save=True)
        phi.capture()
        phi.save(detection=False)
        print("Saved image")
        disp_img = cv2.cvtColor(phi.image,cv2.COLOR_BGR2RGB) # edged_image if using detections
        disp_img = cv2.resize(disp_img, (640,480))
        pic_image = Picture(app, image=Image.fromarray(disp_img), grid=[1,3])
        del phi
    else:
        app.error(title='Error', text='Is location and date set?')
     
def show_video():
    print("Warming up")
    phv = PicamHandler(setting='video', currdir_full=currdir_full)
    time.sleep(1) # warm-up time
    for i, frame in enumerate(phv.camera.capture_continuous(phv.rawCapture, format='bgr', use_video_port=True)):
        print(f'frame: {i}')
        image = frame.array
        image = cv2.resize(image, (640*2,480*2))
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        phv.rawCapture.truncate(0)

        if key == ord("q") or not stop_video_button.enabled:
            phv.camera.close()
            cv2.destroyAllWindows()
            del phv
            break

# ------------- END CAMERA FUNCTIONS -------------
# ------------------------------------------------

#-----------------------------------------------
############ START GUI #########################
def start_video():
    start_video_button.disable()
    stop_video_button.enable()
    t2 = threading.Thread(target=show_video)
    t2.start()

def stop_video():
    start_video_button.enable()
    stop_video_button.disable()

def camera_preview():
    if stop_video_button.enabled:
        if yesno("Video Interruption","You can interrupt video and then take a picture.\nInterrupt video?"):
            stop_video()
    php = PicamHandler(setting='image', currdir_full=currdir_full)
    php.preview(seconds=5)
    del php

def do_on_close():
    if yesno("Close", "Are you sure you want to quit?"):
        app.destroy()

def get_folder():
    madedir_str.value = app.select_folder(folder=default_ses_path)

    if madedir_str.value == default_ses_path:
        app.error(title='Error', text='Session path needs to be a subfolder.')
        madedir_str.value = None
    elif not madedir_str.value.startswith(default_ses_path):
        app.error(title='Error', text='Session path needs to be inside the default sessions folder.')
        madedir_str.value = None
    else:
        global currdir_full
        currdir_full = madedir_str.value
        madedir_str.value = f"Current session: {madedir_str.value.split('/')[-1]}"

def create_sess():    
    name = app.question("Session folder", "Give a name for the session.")
    if name is not None:
        created_experiment = f'{default_ses_path}/{name}'
        make_dirs([created_experiment])
        if not os.path.isdir(default_ses_path):
            os.mkdir(default_ses_path)

        madedir_str.value = created_experiment

        if madedir_str.value == default_ses_path:
            app.error(title='Error', text='Session path needs to be a subfolder inside "sessions", e.g. sessions/test1/')
            madedir_str.value = None
        elif not madedir_str.value.startswith(default_ses_path):
            app.error(title='Error', text='Session path needs to be inside the sessions folder.')
            madedir_str.value = None
        else:
            global currdir_full
            currdir_full = madedir_str.value
            madedir_str.value = f"Current session: {madedir_str.value.split('/')[-1]}"

def select_location():
    print(plateloc_bt.value)

def validate(date_text):
    try:
        datetime.strptime(date_text, '%Y%m%d')
    except:
        print("Incorrect data format, should be YYYYMMDD")
        return False
    return True

def select_date():
    givendate = app.question("Plate date", "Provide the plate\'s date (e.g. w34 or YYYYMMDD like 20201125).")
    if (givendate.startswith('w') and len(givendate) == 3) or validate(givendate):
        if givendate is not None:
            platedate_str.value = givendate
    else:
        app.error(title='Error', text='Date needs to be either in week format: w20 or YYYYMMDD like 20191218.')

# ------------- STOP GUI -------------
# ------------------------------------

# ------------------------------------
# ------------- MAIN -----------------
if __name__=="__main__":

    setup_lights()

    app = App(title="SWD Fly trap v0.2", layout="grid", width=width, height=height, bg = background)

    sess = PushButton(app, command=get_folder, text="Select current session folder", grid=[0,0], align='right')
    makedir_bt = PushButton(app, command=create_sess, text="Create new session folder", grid=[1,0], align='right')
    madedir_str = Text(app, grid=[1,0], align='left')
    
    plateloc_bt = ButtonGroup(app, options=["herent", "kampen", "beauvech", "brainelal", "other"], 
                                    command=select_location, grid=[0,2],
                                    selected="other", align='left')
    platedate_bt = PushButton(app, text='date', command=select_date, grid=[0,2], align='right')
    platedate_str = Text(app, grid=[1,2], align='left')

    pic_image = Picture(app, image=Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0)), grid=[1,3])

    snap_button = PushButton(app, command=snap_detect, text="Take a picture", grid=[0,3], align='left')
    start_video_button = PushButton(app, command=start_video, text="Live Video", grid=[0,4], align='left')
    stop_video_button = PushButton(app, command=stop_video, text="Stop Video", enabled=False, grid=[0,4], align='right')

    # shortpreview_bt = PushButton(app, command=camera_preview, text="Camera preview", grid=[0,6], align='left')

    Text(app, text="Click below to quit and end the session.", grid=[1,6], align='right')
    PushButton(app, command=do_on_close, text='END SESSION', grid=[1,7], align='right')

    app.on_close(do_on_close)
    app.display()
