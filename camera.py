from PIL import Image
# from picamera import PiCamera
# from picamera.array import PiRGBArray
# from picamera import PiCamera
from configparser import ConfigParser
from common import config_path
import time
import logging
import cv2
import pandas as pd

logger = logging.getLogger(__name__)

config = ConfigParser()
config.read(config_path)

camera = config.get('base', 'camera') 
assert camera in ['camera8MP', 'camera12MP'], f"Camera setting not recognized: {camera}"

res_width = int(config.get(camera, 'width'))
res_height = int(config.get(camera, 'height'))


class CameraHandler(object):
    def __init__(self) -> None:
        self.camera = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)
   
        self.unsaved_capture = False
        logger.info("Warming up camera..")
        time.sleep(.5)
        self.camera.read()
        print(f"Resolution defined in settings:\t ({res_width}, {res_height})")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
        print("Auto white-balance enabled.")
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.resolution = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera resolution was set to:\t {self.resolution}")
        # self.camera.release()
        time.sleep(.5)

    def get_possible_resolutions(self):
        url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
        table = pd.read_html(url)[0]
        table.columns = table.columns.droplevel()
        resolutions = {}
        for index, row in table[["W", "H"]].iterrows():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
            width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            resolutions[str(width)+"x"+str(height)] = "OK"
        print(resolutions)

    def capture(self):
        logger.info("Capturing image with camera sensor..")
        assert hasattr(self, 'camera'), "Camera not initialized."

        self.ret, self.frame = self.camera.read()
        if not self.ret:
            print("Failed to capture image..")
            return None

        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.frame)
        logger.info("Image captured.")

        logger.info("Releasing camera..")
        self.camera.release()
        logger.info("Camera released.")
        self.unsaved_capture = True

    def save(self, path):
        logger.info("Saving image..")
        assert self.unsaved_capture, "There is no captured image to save."

        self.path = path
        self.pil_image.save(path)
        logger.info("Image saved.")


# class CameraHandler(object):

#     def __init__(self, resolution=(res_width, res_height), ):
#         logger.info("Initializing camera object..")
#         self.camera = PiCamera()
#         self.camera.awb_mode = 'auto'
#         self.camera.resolution = resolution
#         self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
#         self.unsaved_capture = False
        # logger.info("Warming up camera..")
        # time.sleep(.5)

#     def capture(self):
#         logger.info("Capturing image with camera sensor..")
#         assert hasattr(self, 'camera'), "Camera not initialized."

#         self.camera.capture(self.rawCapture, format="rgb")
#         self.image = self.rawCapture.array
#         self.pil_image = Image.fromarray(self.image)
#         logger.info("Image captured.")
#         logger.info("Closing camera..")
#         self.camera.close()
        # self.unsaved_capture = True
#         logger.info("Closed camera.")

    # def save(self, path):
    #     logger.info("Saving image..")
    #     assert self.unsaved_capture, "There is no captured image to save."

    #     self.path = path
    #     self.pil_image.save(path)
    #     logger.info("Image saved.")
