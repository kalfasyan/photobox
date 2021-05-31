import os
from configparser import ConfigParser

config = ConfigParser()
config_path = './config.ini'
config.read(config_path)

# APP DIMENSIONS
appwidth = int(config.get('app', 'width'))
appheight = int(config.get('app', 'height'))
background = config.get('app', 'backgroundcolor')
blankimgheight = int(config.get('app', 'blankimgheight'))
blankimgwidth = int(config.get('app', 'blankimgwidth'))

default_ses_path = str(config.get('app', 'default_ses_path'))
default_log_path = str(config.get('app', 'default_log_path'))
default_cal_path = str(config.get('app', 'default_cal_path'))
default_prj_path = str(config.get('app', 'default_prj_path'))

for p in [default_cal_path, default_log_path, default_ses_path]:
    if not os.path.exists(p):
        os.makedirs(p)

confidence_threshold = int(config.get('app','confidence_threshold'))
dht22_pin = int(config.get('dht22', 'pin'))
if str(config.get('dht22', 'installed')) == "True":
    dht22_sensor = True
else:
    dht22_sensor = False

with open("LOCATIONS.txt", "r") as f:
    platelocations = f.read().split('\n')

with open("INSECTS.txt", "r") as f:
    insectoptions = f.read().split('\n')

with open("CRITICAL_INSECTS.txt", "r") as f:
    critical_insects = f.read().split('\n')