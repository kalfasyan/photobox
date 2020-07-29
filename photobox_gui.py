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

def update_calib_status():
    global currdir_full
    imgdir = f"{currdir_full}/images/"
    antdir = f"{currdir_full}/annotations/"
    chessboard_imgs_in_currdir = glob.glob(f'{imgdir}/calibration_chessboard*.jpg')
    color_img_in_currdir = glob.glob(f'{imgdir}/calibration_color*.jpg')
    calib_chess_st.value = f"Calib chess: {len(chessboard_imgs_in_currdir)}"
    calib_color_st.value = f"Calib color: {len(color_img_in_currdir)}"
    madedir_str.value = currdir_full

def check_calib_done():
    global currdir_full
    imgdir = f"{currdir_full}/images/"
    antdir = f"{currdir_full}/annotations/"
    
    make_dirs([imgdir, antdir])
    logger.info(f"Looking for calibration plates in {currdir_full}")
    if plateloc_bt.value not in ["other", "calibration_chessboard", "calibration_color"]:
        chessboard_imgs_in_currdir = glob.glob(f'{imgdir}/calibration_chessboard*.jpg')
        color_img_in_currdir = glob.glob(f'{imgdir}/calibration_color*.jpg')

        calib_chess_st.value = f"Calib chess: {len(chessboard_imgs_in_currdir)}"
        calib_color_st.value = f"Calib color: {len(color_img_in_currdir)}"

        logger.info(f"Chessboard images: {len(chessboard_imgs_in_currdir)}")
        logger.info(f"Colorplate image: {len(color_img_in_currdir)}")

        if len(chessboard_imgs_in_currdir) < 10 or len(color_img_in_currdir) < 1:
            app.error(title='Error', text=f'Please perform calibration first. Minimum of 10 chessboard images and one Color plate image. \
                                            Found {len(chessboard_imgs_in_currdir)} chessboard images and {len(color_img_in_currdir)} color plate(s).')
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

def get_folder():
    logger.info("Select session folder button pressed")
    madedir_str.value = app.select_folder(folder=default_ses_path)

    default_ses_path_parent = os.path.abspath(os.path.join(default_ses_path, os.pardir))
    if madedir_str.value == default_ses_path_parent:
        app.error(title='Error', text='Session path needs to be a subfolder.')
        madedir_str.value = None
    elif not madedir_str.value.startswith(default_ses_path_parent):
        app.error(title='Error', text='Session path needs to be inside the default sessions folder.')
        madedir_str.value = None
    else:
        global currdir_full
        currdir_full = madedir_str.value
        imgdir = f"{currdir_full}/images/"
        antdir = f"{currdir_full}/annotations/"
        make_dirs([imgdir, antdir])
        madedir_str.value = f"Current session: {madedir_str.value.split('/')[-1]}"

def create_sess():
    logger.info("Create session button pressed")
    name = app.question("Session folder", "Give a name for the session.")
    if name is not None:
        created_experiment = f'{default_ses_path}/{name}'
        logger.info(f"Created path: {created_experiment}")
        make_dirs([created_experiment])
        if not os.path.isdir(default_ses_path):
            os.mkdir(default_ses_path)

        madedir_str.value = created_experiment
        default_ses_path_parent = os.path.abspath(os.path.join(default_ses_path, os.pardir))
        if madedir_str.value == default_ses_path_parent:
            app.error(title='Error', text='Session path needs to be a subfolder inside "sessions", e.g. sessions/test1/')
            madedir_str.value = None
        elif not madedir_str.value.startswith(default_ses_path_parent):
            app.error(title='Error', text='NOTE: Session path needs to be inside the sessions folder.')
            madedir_str.value = None
        else:
            global currdir_full
            currdir_full = madedir_str.value
            imgdir = f"{currdir_full}/images/"
            antdir = f"{currdir_full}/annotations/"
            
            make_dirs([imgdir, antdir])
            madedir_str.value = f"Current session: {madedir_str.value.split('/')[-1]}"

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

    app = App(title="Photobox v0.1", layout="grid", width=width, height=height, bg = background)

    sess = PushButton(app, command=get_folder, text="Select current session folder", grid=[0,0], align='right')
    makedir_bt = PushButton(app, command=create_sess, text="Create new session folder", grid=[1,0], align='right')
    madedir_str = Text(app, grid=[1,0], align='left')
    
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
