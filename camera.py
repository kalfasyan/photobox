from PIL import Image
from configparser import ConfigParser
from common import config_path
import time
import logging
import cv2
import pandas as pd

logger = logging.getLogger(__name__)

config = ConfigParser()
config.read(config_path)

camera_setting = config.get('base', 'camera') 
assert camera_setting in ['camera8MP', 'camera12MP'], f"Camera setting not recognized: {camera}"

res_width = int(config.get(camera_setting, 'width'))
res_height = int(config.get(camera_setting, 'height'))


class CameraHandler(object):
    def __init__(self) -> None:
        self.camera = cv2.VideoCapture(0)#,cv2.CAP_DSHOW)
   
        self.unsaved_capture = False
        logger.info("Warming up camera..")
        time.sleep(1.)
        self.camera.read()
        print(f"Resolution defined in settings:\t ({res_width}, {res_height})")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
        print("Enabling auto white-balance.")
        self.camera.set(cv2.CAP_PROP_AUTO_WB, 1)
        self.resolution = (int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera resolution was set to:\t {self.resolution}")
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
        # self.camera.release()
        logger.info("Camera released.")
        self.unsaved_capture = True

    def save(self, path):
        logger.info("Saving image..")
        assert self.unsaved_capture, "There is no captured image to save."

        self.path = path
        self.pil_image.save(path)
        logger.info("Image saved.")
