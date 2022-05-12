import os
from pathlib import Path
from configparser import ConfigParser

config = ConfigParser()
config_path = './config.ini'
config.read(config_path)

# SETTING DIRECTORIES
home_path = str(Path().home())
curr_path = str(Path().absolute())
default_ses_path = f"{home_path}/Desktop/PhotoBox/sessions"
default_log_path = f"{home_path}/Desktop/PhotoBox/logs"
default_cal_path = f"{curr_path}/calib"
default_prj_path = curr_path
default_deb_path = f"{home_path}/Desktop/PhotoBox/debugging"
# CREATING DIRECTORIES
for p in [default_cal_path, default_log_path, default_ses_path, default_deb_path]:
    if not os.path.exists(p):
        os.makedirs(p)

# APP DIMENSIONS
appwidth = int(config.get('app', 'width'))
appheight = int(config.get('app', 'height'))
background = config.get('app', 'backgroundcolor')
blankimgheight = int(config.get('app', 'blankimgheight'))
blankimgwidth = int(config.get('app', 'blankimgwidth'))

# APP SETTINGS
confidence_threshold    = float(config.get('app','confidence_threshold'))
minconf_threshold       = float(config.get('app','minconf_threshold'))
modelname               = config.get('app', 'modelname')
crop_pxls_top           = int(config.get('app', 'crop_pxls_top'))
crop_pxls_bot           = int(config.get('app', 'crop_pxls_bot'))
crop_pxls_left          = int(config.get('app', 'crop_pxls_left'))
crop_pxls_right         = int(config.get('app', 'crop_pxls_right'))
bnw_threshold           = int(config.get('app', 'bnw_threshold'))
min_obj_area            = int(config.get('app', 'min_obj_area'))
max_obj_area            = int(config.get('app', 'max_obj_area'))
nms_threshold           = float(config.get('app', 'nms_threshold'))

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
critical_insects = list(filter(None, critical_insects))
