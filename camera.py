import cv2
from PIL import Image
import picamera
from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera
from configparser import ConfigParser
from common import config_path, make_dirs, check_dir_location, platelocations
import time
import logging

logger = logging.getLogger(__name__)

config = ConfigParser()
config.read(config_path)

res_width = int(config.get('camera', 'width'))
res_height = int(config.get('camera', 'height'))

class CameraHandler(object):

    def __init__(self, resolution=(res_width, res_height), ):
        logger.info("Initializing camera object..")
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        self.unsaved_capture = False
        logger.info("Warming up camera..")
        time.sleep(.5)

    def capture(self):
        logger.info("Capturing image with camera sensor..")
        assert hasattr(self, 'camera'), "Camera not initialized."

        self.camera.capture(self.rawCapture, format="rgb")
        self.image = self.rawCapture.array
        self.pil_image = Image.fromarray(self.image)
        logger.info("Image captured.")
        logger.info("Closing camera..")
        self.camera.close()
        self.unsaved_capture = True
        logger.info("Closed camera.")

    def save(self, path):
        logger.info("Saving image..")
        assert self.unsaved_capture, "There is no captured image to save."

        self.path = path
        self.pil_image.save(path)
        logger.info("Image saved.")