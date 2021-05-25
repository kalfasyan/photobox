import pathlib, os
from configparser import ConfigParser

def get_latest_filename(path=None):
    allpaths = sorted(pathlib.Path(path).iterdir(), key=os.path.getmtime)
    return f"{path}/{allpaths[-1].name}"

def make_dirs(paths=[]):
    for p in paths:
        if not os.path.exists(p): 
            os.mkdir(p)

def make_session_dirs(curdir='', paths=['images','annotations','detections']):
    dirs = []
    for p in paths:
        dirs.append(f"{curdir}/{p}/")

    make_dirs(dirs)
    return [i for i in dirs]

def check_dir_location(path=None):
    if isinstance(path, str) and path.startswith(default_ses_path) and path != default_ses_path:
        return True
    else:
        return False

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

for p in [default_cal_path, default_log_path, default_ses_path]:
    if not os.path.exists(p):
        os.makedirs(p)

dht22_pin = int(config.get('dht22', 'pin'))
if str(config.get('dht22', 'installed')) == "True":
    dht22_sensor = True
else:
    dht22_sensor = False

with open("LOCATIONS.txt", "r") as f:
    platelocations = f.read().split('\n')

with open("INSECTS.txt", "r") as f:
    insectoptions = f.read().split('\n')