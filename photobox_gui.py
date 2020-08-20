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
imgdir = f"{currdir_full}/images/"
antdir = f"{currdir_full}/annotations/"

make_dirs([imgdir, antdir])

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
    if len(platedate_str.value):
        logger.info(f"Plate date set to: {platedate_str.value}")
        phi = PicamHandler(setting='image', currdir_full=currdir_full, plateloc=plateloc_bt.value ,platedate=platedate_str.value, platenotes=platenotes_str.value)
        # phi.capture_and_detect(save=True)
        phi.capture()
        if not plateloc_bt.value.startswith('calibration'):
            phi.detect()
            disp_img = phi.edged_image
        else:
            disp_img = phi.image
        phi.save(detection=False)
        logger.info("Saved image")

        disp_img = cv2.cvtColor(disp_img,cv2.COLOR_BGR2RGB) # edged_image if using detections otherwise image
        disp_img = cv2.resize(disp_img, (640,480))
        pic_image = Picture(app, image=Image.fromarray(disp_img), grid=[1,2])
        pic_path.value = phi.picpath.split('/')[-1]
        del phi
    else:
        app.error(title='Error', text='Is location and date set?')


# ------------- END CAMERA FUNCTIONS -------------
# ------------------------------------------------

#-----------------------------------------------
############ START GUI #########################

def update_calib_status(nr_calib_plates=False):
    global currdir_full
    imgdir = f"{currdir_full}/images/"
    antdir = f"{currdir_full}/annotations/"
    make_dirs([imgdir, antdir])
    chessboard_imgs_in_currdir = glob.glob(f'{imgdir}/calibration_chessboard*.jpg')
    color_img_in_currdir = glob.glob(f'{imgdir}/calibration_color*.jpg')
    calib_chess_st.value = f"Calib chess: {len(chessboard_imgs_in_currdir)}"
    calib_color_st.value = f"Calib color: {len(color_img_in_currdir)}"
    logger.info(f"Chessboard images: {len(chessboard_imgs_in_currdir)}")
    logger.info(f"Colorplate image: {len(color_img_in_currdir)}")
    selected_sesspath.value = currdir_full

    if nr_calib_plates:
        return len(chessboard_imgs_in_currdir), len(color_img_in_currdir)

def check_calib_done():
    if plateloc_bt.value not in ["other", "calibration_chessboard", "calibration_color"]:
        nr_chess_imgs, nr_color_imgs = update_calib_status(nr_calib_plates=True)
        if nr_chess_imgs < 10 or nr_color_imgs < 1:
            app.error(title='Error', text=f'Please perform calibration first. Minimum of 10 chessboard images and one Color plate image. \
                                            Found {nr_chess_imgs} chessboard images and {nr_color_imgs} color plate(s).')
            return False
        else:
            calib_chess_st.value = f"Chessboard images: OK"
            calib_color_st.value = f"Colorplate images: OK"
    return True

def take_picture():
    logger.info("Taking picture button pressed")
    check_calib = check_calib_done()
    if check_calib:
        snap_detect()
        update_calib_status()

def do_on_close():
    logger.info("Quit button pressed")
    if yesno("Close", "Are you sure you want to quit?"):
        app.destroy()

def check_session_path(name, created_new=False):
    if created_new:
        user_created_sesspath = f'{default_ses_path}/{name}'
    else:
        user_created_sesspath = name

    default_ses_path_parent = os.path.abspath(os.path.join(default_ses_path, os.pardir))

    if len(user_created_sesspath.split('/')) > 6:
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions" and cannot be \'images\' or \'annotations\'')
        return None
    elif user_created_sesspath.endswith('images') or user_created_sesspath.endswith('annotations'):
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions" and cannot be \'images\' or \'annotations\'')
        return None
    elif user_created_sesspath.split('/')[-1] == 'sessions':
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions"')
        return None
    elif user_created_sesspath == default_ses_path_parent:
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions", e.g. sessions/test1/')
        return None
    elif not user_created_sesspath.startswith(default_ses_path_parent):
        app.error(title='Error', text='NOTE: Session path needs to be inside the sessions folder.')
        return None
    else:
        make_dirs([default_ses_path, user_created_sesspath])
        logger.info(f"Created path: {user_created_sesspath}")
        return user_created_sesspath

def get_folder():
    logger.info("Select session folder button pressed")
    name = app.select_folder(folder=default_ses_path)
    selected_sesspath.value = check_session_path(name, created_new=False)

    if selected_sesspath.value is not None:
        global currdir_full
        currdir_full = selected_sesspath.value
        imgdir = f"{currdir_full}/images/"
        antdir = f"{currdir_full}/annotations/"
        make_dirs([imgdir, antdir])
        selected_sesspath.value = f"{selected_sesspath.value.split('/')[-1]}"

def create_sess():
    logger.info("Create session button pressed")
    name = app.question("Session folder", "Give a name for the session.")
    selected_sesspath.value = check_session_path(name, created_new=True)

    if selected_sesspath.value is not None:
        global currdir_full
        currdir_full = selected_sesspath.value
        imgdir = f"{currdir_full}/images/"
        antdir = f"{currdir_full}/annotations/"
        
        make_dirs([imgdir, antdir])
        selected_sesspath.value = f"{selected_sesspath.value.split('/')[-1]}"
        platedate_str.value = ''

def select_location():
    logger.info(f"Selected location: {plateloc_bt.value}")

def validate(date_text):
    try:
        datetime.strptime(date_text, '%Y%m%d')
    except:
        print("Incorrect data format, should be YYYYMMDD")
        return False
    return True

def select_date():
    logger.info("Date button pressed")
    givendate = app.question("Plate date", "Provide the plate\'s date (e.g. w34 or YYYYMMDD like 20201125).")
    if (givendate.startswith('w') and len(givendate) == 3) or validate(givendate):
        if givendate is not None:
            platedate_str.value = givendate
    else:
        app.error(title='Error', text='Date needs to be either in week format: w20 or YYYYMMDD like 20191218.')

def enter_notes():
    logger.info("Notes button pressed")
    givennotes = app.question("Plate notes", "Give some extra location-notes regarding the plate. e.g. 1-60, centroid etc.")
    if len(givennotes) <= 10:
        platenotes_str.value = givennotes
    else:
        app.error(title='Error', text='Length of text provided is too long. Up to 10 characters allowed.')

# ------------- STOP GUI -------------
# ------------------------------------

# ------------------------------------
# ------------- MAIN -----------------
if __name__=="__main__":

    setup_lights()

    app = App(title="Photobox v0.5", layout="grid", width=width, height=height, bg = background)

    sess = PushButton(app, command=get_folder, text="Select current session folder", grid=[0,0], align='right')
    makedir_bt = PushButton(app, command=create_sess, text="Create new session folder", grid=[1,0], align='right')
    selected_sesspath = Text(app, grid=[1,0], align='left')
    
    plateloc_bt = ButtonGroup(app, options=platelocations, 
                                    command=select_location, grid=[0,2],
                                    selected="other", align='left')
    platedate_bt = PushButton(app, text='Plate date', command=select_date, grid=[0,4], align='right')
    platedate_str = Text(app, grid=[1,4], align='left')

    platenotes_bt = PushButton(app, text='extra notes (e.g. 3-60 or \"centroid\")', command=enter_notes, grid=[0,3], align='right')
    platenotes_str = Text(app, grid=[1,3], align='left')

    pic_image = Picture(app, image=Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0)), grid=[1,2])
    
    snap_button = PushButton(app, command=take_picture, text="Take a picture", grid=[0,5], align='right')
    pic_path = Text(app, grid=[1,5], align='left')
    calib_chess_st = Text(app, grid=[0,4], align='left')
    calib_color_st = Text(app, grid=[0,5], align='left')

    # shortpreview_bt = PushButton(app, command=camera_preview, text="Camera preview", grid=[0,6], align='left')

    Text(app, text="Click below to quit and end the session.", grid=[1,7], align='right')
    PushButton(app, command=do_on_close, text='END SESSION', grid=[1,8], align='right')

    app.on_close(do_on_close)
    app.display()
