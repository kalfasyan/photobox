import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2, time, os, glob
import imutils
import numpy as np
from datetime import datetime
from configparser import ConfigParser
import threading
import pathlib
import logging
from common import config_path, make_dirs, check_dir_location

logger = logging.getLogger(__name__)

config = ConfigParser()
config.read(config_path)

############### LIGHTS #######################
LEDpin = int(config.get('lights', 'LEDpin'))
lights_on = config.get('lights', 'withlights') in ['True','true','Y','y','yes','Yes','YES']

def setup_lights(on=lights_on):
    if on:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LEDpin, GPIO.OUT)
        GPIO.output(LEDpin, GPIO.LOW)
    else:
        logger.debug("LIGHTS SETUP: Lights are not enabled.")

def switch_off_lights(on=lights_on):
    if on:
        GPIO.output(LEDpin, GPIO.LOW)
    else:
        logger.debug("LIGHTS OFF: Lights are not enabled.")

def switch_on_lights(on=lights_on):
    if on:
        GPIO.output(LEDpin, GPIO.HIGH)
    else:
        logger.debug("LIGHTS ON: Lights are not enabled.")

def clear_lights(on=lights_on):
    if on:
        GPIO.cleanup()
    else:
        logger.debug("LIGHTS CLEANUP: Lights are not enabled.")
###############################################

class PicamHandler:
    def __init__(self, setting="image", currdir_full=None, plateloc=None, platedate=None):
        assert setting in ['image', 'video'], "Only image or video accepted."
        self.setting = setting
        assert check_dir_location(currdir_full), "Wrong base directory for wingbeat sensor."        
        self.currdir_full = currdir_full
        self.imgdir = f"{currdir_full}/images"
        self.viddir = f"{currdir_full}/videos"
        make_dirs([self.imgdir, self.viddir])
                
        self.plateloc = plateloc
        self.platedate = platedate

        if self.setting == 'image':
            self.width = int(config.get('camera', 'width'))
            self.height = int(config.get('camera', 'height'))
            try:
                self.camera = PiCamera()
                self.camera.resolution = (self.width, self.height)
                self.rawCapture = PiRGBArray(self.camera, size=(self.width, self.height))
            except:
                print("ERROR: Camera unavailable. Maybe another process is using it?")

        elif self.setting== 'video':
            self.width = int(config.get('videocamera', 'width'))
            self.height = int(config.get('videocamera', 'height'))
            self.recording = False
            try:
                self.camera = PiCamera()
                self.camera.resolution = (self.width, self.height)
                self.camera.framerate = int(config.get('videocamera', 'framerate'))
                self.rawCapture = PiRGBArray(self.camera, size=(self.width, self.height))
            except:
                print("ERROR: Camera unavailable. Maybe another process is using it?")

    ### FUNCTIONS FOR IMAGES ###
    def capture(self):
        assert self.setting == 'image', "Function used for images only."
        assert hasattr(self, 'camera'), "Camera not initialized."
        assert self.plateloc in ['herent','kampen','beauvech','brainelal','other'], f"Wrong plate location provided: {self.plateloc}."
        assert self.platedate.startswith('w') or self.platedate.startswith('20') and len(self.platedate) <= 8, "Wrong plate date provided."

        self.camera.capture(self.rawCapture, format="bgr")
        self.image = self.rawCapture.array
        self.camera.close()

    def preview(self, seconds=5):
        assert self.setting == 'image', "Function used for images only."
        assert hasattr(self, 'camera'), "Camera not initialized."
        self.camera.start_preview()
        time.sleep(seconds)
        self.camera.stop_preview()
        self.camera.close()

    def detect(self):
        assert self.setting == 'image', "Function used for images only."
        assert hasattr(self, 'image'), "No image captured yet."
        image = self.image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)#11, 3)
        filtered = cv2.medianBlur(gray,7) # 13
        edged = cv2.Canny(filtered, 30, 150)

        # Highlighting detections
        (cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug("{} insects counted in captured image.".format(len(cnts)))
        edged_image = image.copy()
        cv2.drawContours(edged_image, cnts, -1, (0, 255, 0), 1)
        self.edged_image = edged_image
        self.cnts = cnts

    def capture_and_detect(self, save=True, savedir=None):
        self.capture()
        self.detect()
        if save:
            self.save(savedir=savedir)

    def save(self, detection=True, savedir=None):
        assert self.setting == 'image', "Function used for images only."
        assert hasattr(self, 'image'), "No image captured yet."
        if savedir is None:
            savedir = self.imgdir

        ResultName = f"{savedir}/{self.plateloc}_{self.platedate}_{self.width}x{self.height}_{str(time.strftime('%d%m%Y%H%M%S'))}.jpg"
        cv2.imwrite(ResultName, self.image)
        if detection:
            assert hasattr(self, 'edged_image'), "No detection performed yet. Run detect() first."
            cv2.imwrite(f'{ResultName[:-4]}_detection.jpg', self.edged_image)

    ### FUNCTIONS FOR VIDEO ###
    def _init_video(self, show=False, maxseconds=10):
        assert self.setting == 'video', "Function used for video only."
        assert not self.camera.closed, "Camera is closed, create new object."
        self.recording = True

        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        vidfilename = f"{self.viddir}/Vid_{self.width}x{self.height}_{curr_time}.h264"
        
        self.camera.start_recording(vidfilename)
        self.rec_t0 = datetime.now()
        self.camera.wait_recording(maxseconds)

        for i, frame in enumerate(self.camera.capture_continuous(self.rawCapture, format='bgr', use_video_port=True)):
            image = frame.array
            print(f"frame: {i}, shape: {image.shape}")
            self.rawCapture.truncate(0)

            elapsed = (datetime.now() - self.rec_t0).seconds
            if elapsed > maxseconds or elapsed > 300:
                break

        self.camera.stop_recording()
        self.camera.stop_preview()
        self.camera.close()
        self.recording = False

    def start_rec(self, maxseconds=10):
        self.thr = threading.Thread(target=self._init_video, kwargs={'show': False, 'maxseconds': maxseconds})
        self.thr.start()
