#!/home/pi/.virtualenvs/cv/bin/python3
"""
Author: Ioannis Kalfas
PhD Researcher at MeBioS, KU Leuven
contact: ioannis.kalfas[at]kuleuven.be or kalfasyan[at]gmail.com
"""
import glob
import io
import os
import pathlib
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import picamera
import RPi.GPIO as GPIO
import scipy.integrate as it
from guizero import *
from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image
from scipy.io.wavfile import write
from scipy.signal import savgol_filter

from camera import *
from camera import CameraHandler
from common import *
from snap_detect import *
from stickyplate import StickyPlate, resize_pil_image

currdir_full = f'{default_ses_path}/{datetime.now().strftime("%Y%m%d")}'
if not os.path.isdir(default_ses_path):
    os.mkdir(default_ses_path)
if not os.path.isdir(currdir_full):
    os.mkdir(currdir_full)
caldir = f"{default_cal_path}"
imgdir, antdir, dtcdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections'])

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

def snap():
    cam = CameraHandler()
    w, h = cam.camera.resolution
    full_platepath = Path(f"{imgdir}/TEST.png")

    cam.capture()
    cam.save(full_platepath)

    global sp
    sp = StickyPlate(full_platepath, caldir)
    sp.undistort(inplace=True)

    pic_image = Picture(app, image=resize_pil_image(sp.pil_image, basewidth=620), grid=[1,2])
    pic_path.value = full_platepath

def segment():
    try:
        global sp
    except:
        logger.info("Take a picture first.")
    assert sp.undistorted
    sp.threshold_image(threshold=127)

    pic_image = Picture(app, image=resize_pil_image(sp.pil_thresholded, basewidth=620), grid=[1,2])

def crop():
    try:
        global sp
    except:
        logger.info("Take a picture first.")
    assert sp.undistorted
    assert not sp.cropped, "Already cropped"
    sp.crop_image(height_pxls=100, width_pxls=120)

    pic_image = Picture(app, image=resize_pil_image(sp.pil_image, basewidth=620), grid=[1,2])

def detect():
    try:
        global sp
    except:
        logger.info("Take a picture first.")

    assert sp.undistorted
    assert sp.segmented, "Segment image first"
    
    sp.detect_objects(min_obj_area=15, max_obj_area=6000, nms_threshold=0.08, insect_img_dim=150)
    sp.save_detections(savepath=dtcdir)        

    pic_image = Picture(app, image=resize_pil_image(sp.pil_image_bboxes, basewidth=620), grid=[1,2])

def snap_detect():
    if len(platedate_str.value):
        logger.info(f"Plate date set to: {platedate_str.value}")

        cam = CameraHandler()
        w, h = cam.camera.resolution
        plateloc = plateloc_bt.value
        plateinfo = platenotes_str.value if len(platenotes_str.value) else "NA"
        platedate = platedate_str.value
        platename = f"{plateloc}_{plateinfo}_{platedate}_{w}x{h}.png"
        full_platepath = Path(f"{imgdir}/{platename}")

        cam.capture()
        cam.save(full_platepath)

        global sp
        sp = StickyPlate(full_platepath, caldir)
        sp.undistort(inplace=True)
        sp.crop_image()
        sp.threshold_image()

        if not plateloc_bt.value.startswith('calibration'):
            sp.detect_objects()
            sp.save_detections(savepath=dtcdir)
            disp_img = sp.pil_image_bboxes
        else:
            disp_img = sp.pil_image

        logger.info("Saved image")

        pic_image = Picture(app, image=resize_pil_image(disp_img, basewidth=620), grid=[1,2])
        pic_path.value = full_platepath
        del cam
    else:
        app.error(title='Error', text='Is location and date set?')


# ------------- END CAMERA FUNCTIONS -------------
# ------------------------------------------------

#-----------------------------------------------
############ START GUI #########################

def update_calib_status(nr_calib_plates=False):
    global currdir_full

    chessboard_imgs_in_currdir = glob.glob(f'{caldir}/calibration_chessboard*.jpg')
    color_img_in_currdir = glob.glob(f'{caldir}/calibration_color*.jpg')
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
    check_calib = True # check_calib_done() 
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
        selected_sesspath.value = f"{selected_sesspath.value.split('/')[-1]}"
        global imgdir, antdir, dtcdir
        imgdir, antdir, dtcdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections'])

def create_sess():
    logger.info("Create session button pressed")
    name = app.question("Session folder", "Give a name for the session.")
    selected_sesspath.value = check_session_path(name, created_new=True)

    if selected_sesspath.value is not None:
        global currdir_full
        currdir_full = selected_sesspath.value
        selected_sesspath.value = f"{selected_sesspath.value.split('/')[-1]}"
        platedate_str.value = ''
        global imgdir, antdir, dtcdir
        imgdir, antdir, dtcdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections'])

def open_currdir():
    import subprocess
    global currdir_full
    p = subprocess.Popen(["pcmanfm", "%s" % f"{currdir_full}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_imgdir():
    import subprocess
    global imgdir
    p = subprocess.Popen(["pcmanfm", "%s" % f"{imgdir}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_dtcdir():
    import subprocess
    global dtcdir
    p = subprocess.Popen(["pcmanfm", "%s" % f"{dtcdir}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

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

def open_validation_window():
    global dtcdir, detections_list, val_idx
    val_idx = 0
    detections_list = [str(fname) for fname in pathlib.Path(dtcdir).glob('**/*.png')]
    val_img.image = detections_list[val_idx]

    val_window.show()

def close_validation_window():
    val_window.hide()

def next_val():
    global dtcdir, detections_list, val_idx
    print(val_idx)
    if val_idx < len(detections_list)-1:
        val_idx+=1
        val_img.image = detections_list[val_idx]
    else:
        val_idx = 0
        val_img.image = detections_list[val_idx]

def prev_val():
    global dtcdir, detections_list, val_idx
    print(val_idx)
    if val_idx > 0:
        val_idx-=1
        val_img.image = detections_list[val_idx]
    else:
        val_idx = len(detections_list)-1
        val_img.image = detections_list[val_idx]

# ------------- STOP GUI -------------
# ------------------------------------

# ------------------------------------
# ------------- MAIN -----------------
if __name__=="__main__":

    setup_lights()

    app = App(title="Photobox v0.7", layout="grid", width=width, height=height, bg = background)

    # MENU
    menubar = MenuBar(app, 
                    toplevel=["File","Open"],
                    options=[
                        [ ["Change current session..", get_folder], ["New session..", create_sess] ],
                        [ ["Open session directory..", open_currdir], 
                            ["Open image directory..", open_imgdir], 
                            ["Open detections directory..", open_dtcdir],],
                            ])
    # MAIN INTERFACE
    sess                = Text(app, grid=[0,0], align='right')
    sess.value          = 'Current session:'
    selected_sesspath   = Text(app, grid=[1,0], align='left')
    plateloc_bt         = ButtonGroup(app, options=platelocations, 
                                    command=select_location, grid=[0,2],
                                    selected="other", align='left')
    platenotes_bt       = PushButton(app, text='INFO', command=enter_notes, grid=[0,3], align='right')
    platenotes_str      = Text(app, grid=[1,3], align='left')
    platedate_bt        = PushButton(app, text='DATE', command=select_date, grid=[0,4], align='right')
    platedate_str       = Text(app, grid=[1,4], align='left')
    pic_image           = Picture(app, image=Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0)), grid=[1,2])
    last_img            = Text(app, grid=[0,5], align='right')
    last_img.value      = "LAST IMAGE:"
    snapdetect_button   = PushButton(app, command=take_picture, text="FULL RUN", grid=[1,6], align='top')
    pic_path            = Text(app, grid=[1,5], align='left')
    stats_box           = Box(app, height='fill', align='left', grid=[2,3])
    calib_chess_st      = Text(stats_box, align='top')
    calib_color_st      = Text(stats_box, align='top')
    imgproc_box         = Box(app, height="fill", align="left", grid=[2,2])
    snap_button         = PushButton(imgproc_box, command=snap, text="SNAP", align='top')
    crop_button         = PushButton(imgproc_box, command=crop, text="CROP", align='top')
    segment_button      = PushButton(imgproc_box, command=segment, text="SEGMENT", align='top')
    detect_button       = PushButton(imgproc_box, command=detect, text="DETECT", align='top')

    # VALIDATION WINDOW
    val_window          = Window(app, visible=False, bg='white', title="VALIDATE DETECTIONS")
    openval_button      = PushButton(app, text="VALIDATE", command=open_validation_window, grid=[2,7])
    closeval_button     = PushButton(val_window, text="END", command=close_validation_window, align='top')
    val_img             = Picture(val_window, image=Image.new('RGB', (blankimgwidth//2, blankimgheight//2), (0,0,0)), align='top')
    next_val_button     = PushButton(val_window, command=next_val, text="NEXT", align='top')
    prev_val_button     = PushButton(val_window, command=prev_val, text="PREVIOUS", align='top')

    app.on_close(do_on_close)
    app.display()
