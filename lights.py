import logging
import RPi.GPIO as GPIO
from configparser import ConfigParser
from common import config_path

config = ConfigParser()
config.read(config_path)
logger = logging.getLogger(__name__)


LEDpin = int(config.get('lights', 'LEDpin'))
lights_on = config.get('lights', 'managedlights') in ['True','true','Y','y','yes','Yes','YES']

def setup_lights(on=lights_on):
    if on:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(LEDpin, GPIO.OUT)
        GPIO.output(LEDpin, GPIO.LOW)
    else:
        logger.debug("Commanded to SETUP LIGHTS: Managed lights are not enabled.")

def switch_off_lights(on=lights_on):
    if on:
        GPIO.output(LEDpin, GPIO.LOW)
    else:
        logger.debug("Commanded to switch LIGHTS OFF: Managed lights are not enabled.")

def switch_on_lights(on=lights_on):
    if on:
        GPIO.output(LEDpin, GPIO.HIGH)
    else:
        logger.debug("Commanded to switch LIGHTS ON: Managed lights are not enabled.")

def clear_lights(on=lights_on):
    if on:
        GPIO.cleanup()
    else:
        logger.debug("Commanded to do LIGHTS CLEANUP: Managed lights are not enabled.")