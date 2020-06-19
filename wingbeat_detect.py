import audioop
import glob
import logging
import multiprocessing
import os
import threading
import time
import warnings
import wave
from collections import deque
from configparser import ConfigParser
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from scipy.io.wavfile import write

import pyaudio
import RPi.GPIO as GPIO
from common import (blankimgheight, blankimgwidth, check_dir_location,
                    config_path, dht22_pin, dht22_sensor, make_dirs)
from snap_detect import PicamHandler

config = ConfigParser()

logger = logging.getLogger(__name__)

############ LIGHTS ###################
config.read(config_path)

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
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

class WingbeatSensorHandler:

    def __init__(self, obj_detect=False, show_detections=False, currdir_full=None):
        assert check_dir_location(currdir_full), "Wrong base directory for wingbeat sensor."
        self.currdir_full = currdir_full
        self.wbtdir = f"{currdir_full}/wingbeats"
        self.sctdir = f"{currdir_full}/sensor_camera_triggers"
        make_dirs([self.sctdir, self.wbtdir])


        self.thresh = int(config.get('stream', 'thresh'))
        self.chunk = int(config.get('stream', 'chunk'))
        self.format = pyaudio.paFloat32
        self.channels = int(config.get('stream', 'channels'))
        self.rate = int(config.get('stream', 'rate'))
        self.device = int(config.get('stream', 'device'))
        self.buffersize = int(config.get('stream', 'buffersize'))
        self.show_detections = show_detections
        self.obj_detect = obj_detect
        self.light = False
        self.newplot = False
        self.plot = np.ones((blankimgheight, blankimgwidth))#Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0))
        self.temperature = 'na'
        self.humidity = 'na'

    def initialize(self, show_detections=False):
        self.bufferlist = deque(maxlen=self.buffersize)
        self.p = pyaudio.PyAudio()
        dev = int(self.device)

        self.stream = self.p.open(format=self.format,
                                    channels=self.channels,
                                    rate=self.rate,
                                    input_device_index=dev,
                                    input=True,
                                    frames_per_buffer=self.chunk)            
        self.t1 = datetime.now()
        if self.show_detections:
            self.fig = Figure()
            self.canvas = FigureCanvas(self.fig)
            self.ax = self.fig.gca()

    def read(self, detect=False):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        rms = audioop.rms(data, 2)
        logging.debug(f"RMS: {rms}")
        
        self.bufferlist.append( np.frombuffer(data, 'Float32') )

        if rms > self.thresh:
            switch_on_lights()
            self.light = True
            self.t1 = datetime.now()
            logger.info("signal detected")

            curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.obj_detect:
                camthread = multiprocessing.Process(target=iterate_capture_detect, 
                                            kwargs={'sctdir': self.sctdir,
                                                    'currdir_full': self.currdir_full,
                                                    'curr_time': self.t1,
                                                    'maxseconds': 5})
                camthread.start()
                # camthread.join(timeout=5)
                # camthread.terminate()

            for t in range(self.buffersize-1):
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.bufferlist.append(np.frombuffer(data, 'Float32'))
            
            raw_signal = np.hstack(self.bufferlist)

            if dht22_sensor:
                self.temperature, self.humidity = get_temperature_humidity(secs_to_try=2)

            write(f'{self.wbtdir}/{curr_time}_T{self.temperature}_H{self.humidity}.wav', self.rate, raw_signal)

            if self.show_detections:
                self.ax.plot(raw_signal)
                self.canvas.draw()
                sss, (width, height) = self.canvas.print_to_buffer()
                self.plot = np.fromstring(sss, np.uint8).reshape((height, width, 4))
                self.newplot = True
                self.ax.clear()

        if (datetime.now() - self.t1).seconds > 10 and self.light:
            switch_off_lights()
            self.light = False

    def finalize(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        clear_lights()

def get_temperature_humidity(secs_to_try=2):
    import Adafruit_DHT

    t0 = datetime.now()
    elapsed_secs = 0
    humidity, temperature = 'na', 'na'

    while 'na' in [humidity, temperature] and elapsed_secs < secs_to_try:
        humidity, temperature = Adafruit_DHT.read(Adafruit_DHT.DHT22, dht22_pin)
        elapsed_secs = (datetime.now() - t0).seconds

    return temperature, humidity

def iterate_capture_detect(curr_time=None, maxseconds=1, sctdir=None, currdir_full=None):
    ph = PicamHandler(setting='image', currdir_full=currdir_full)
    elapsed_seconds = (datetime.now() - curr_time).seconds

    while elapsed_seconds < maxseconds:
        try:
            elapsed_seconds = (datetime.now() - curr_time).seconds
            time.sleep(1.)
            ph.capture(keep_open=True)
            logger.debug("iterate_capture_detect: Captured image")
            logger.debug(f"Time passed since call: {elapsed_seconds}")
            ph.rawCapture.truncate(0)
            ph.detect()
            ph.save(detection=True, savedir=sctdir)            
        except:
            logger.debug("Camera busy")
            continue

        timenow = datetime.now().strftime("%Y%m%d_%H%M%S")

        with open(f'{sctdir}/bounding_box_centroids.txt', 'a') as f:
            for c in ph.cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                try:
                    x = int(M["m10"] / M["m00"])
                except:
                    x = 'nan'
                try:
                    y = int(M["m01"] / M["m00"])
                except:
                    y = 'nan'
                logger.debug(f'Centroid: {timenow}_{x}_{y}')
                f.write(f'centroid_{timenow}_{x}_{y}\n')

    logger.debug("Closing the camera after iterate_capture_detect")
    ph.camera.close()
    del ph
    return
