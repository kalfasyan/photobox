import pathlib, os
from configparser import ConfigParser

def get_latest_filename(path=None):
    allpaths = sorted(pathlib.Path(path).iterdir(), key=os.path.getmtime)
    return f"{path}/{allpaths[-1].name}"

def make_dirs(paths=[]):
    for p in paths:
        if not os.path.exists(p): 
            os.mkdir(p)

def check_dir_location(path=None):
    if isinstance(path, str) and path.startswith(default_ses_path) and path != default_ses_path:
        return True
    else:
        return False

config = ConfigParser()
config_path = './config.ini'
config.read(config_path)

width = int(config.get('app', 'width'))
assert 1000 > width > 500, "App window width dimension error! Change settings"
height = int(config.get('app', 'height'))
assert 1300 > height > 500, "App window height dimension error! Change settings"
background = config.get('app', 'backgroundcolor')
assert background in ['white','black','red','green','blue','yellow'], "Error! Wrong background color given in settings"
blankimgheight = int(config.get('app', 'blankimgheight'))
assert 800 > blankimgheight > 100, "Error! Change blankimgheight dimensions in settings"
blankimgwidth = int(config.get('app', 'blankimgwidth'))
assert 800 > blankimgwidth > 100, "Error! Change blankimgwidth dimensions in settings"

default_ses_path = str(config.get('app', 'default_ses_path'))

dht22_pin = int(config.get('dht22', 'pin'))
if str(config.get('dht22', 'installed')) == "True":
    dht22_sensor = True
else:
    dht22_sensor = False

platelocations = [
                "calibration_chessboard",
                "calibration_color",
                "kampen", 
                "beauvech", 
                "brainelal",
                "herent", 
                "herentVAL1", 
                "herentVAL2",
                "herentVAL3",
                "herentVAL4",
                "herentVAL5",
                "herentCONTROLE",
                "herentVALdelta",
                "other"]