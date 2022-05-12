#!/home/pi/.virtualenvs/cv/bin/python3
"""
Author: Ioannis Kalfas
PhD Researcher at MeBioS, KU Leuven
contact: ioannis.kalfas[at]kuleuven.be or kalfasyan[at]gmail.com
"""
import glob
import os
import pathlib
import shutil
import re
import warnings
from datetime import datetime
from pathlib import Path
from natsort import natsorted
import subprocess
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from guizero import *
from PIL import Image
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.4)

from camera import *
from common import *
from stickyplate import StickyPlate, resize_pil_image
from detections import *
from utils import get_cpu_temperature, make_session_dirs, make_dirs
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

global cpu_temperature, insectoptions, confidence_threshold
cpu_temperature = 'NA'

global default_log_path, default_ses_path, default_cal_path, caldir, default_prj_path,modelname, debdir
currdir_full = f'{default_ses_path}/{datetime.now().strftime("%Y%m%d")}'
if not os.path.isdir(default_ses_path):
    os.mkdir(default_ses_path)
if not os.path.isdir(currdir_full):
    os.mkdir(currdir_full)
caldir = f"{default_cal_path}"
debdir = f"{default_deb_path}"
imgdir, antdir, dtcdir, expdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections','exports'])

warnings.simplefilter("once", DeprecationWarning)

# -------------------------------------------------
# ------------- LOGGER CONFIG ---------------------

logdate = datetime.now().strftime("%Y%m%d")
streamhandler = logging.StreamHandler()
streamhandler.terminator = "\n"
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[
                        RotatingFileHandler(f'{default_log_path}/photobox_logs_{logdate}.log',
                                            maxBytes=50000000, backupCount=50),
                        streamhandler])

logger = logging.getLogger(__name__)

#---------------------------------------------------
# ------------- START CAMERA FUNCTIONS -------------

def snap():

    cam = CameraHandler()
    w,h = cam.resolution

    plateloc = plateloc_bt.value
    plateinfo = platenotes_str.value if len(platenotes_str.value) else "NA"
    platedate = platedate_str.value
    platename = f"{plateloc}_{plateinfo}_{platedate}_{w}x{h}.png"
    full_platepath = Path(f"{imgdir}/{platename}")
    cam.capture()
    cam.save(full_platepath)

    global sp
    sp = StickyPlate(full_platepath, caldir)

    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(sp.image), basewidth=blankimgwidth-100), grid=[1,2])
    pic_path.value = full_platepath

def calibrate():
    try:
        global sp
    except:
        logger.info("Take a picture first.")
    sp.undistort(inplace=True)
    # sp.colorcorrect(inplace=True)

    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(sp.image), basewidth=blankimgwidth-100), grid=[1,2])

def segment():
    try:
        global sp
    except:
        logger.info("Take a picture first.")
    assert sp.undistorted
    sp.threshold_image()

    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(sp.thresholded), basewidth=blankimgwidth-100), grid=[1,2])

def crop():
    try:
        global sp
    except:
        logger.info("Take a picture first.")
    assert sp.undistorted
    assert not sp.cropped, "Already cropped"
    sp.crop_image()

    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(sp.image), basewidth=blankimgwidth-100), grid=[1,2])

def detect():
    try:
        global sp
    except:
        logger.info("Take a picture first.")

    assert sp.undistorted
    assert sp.segmented, "Segment image first"
    
    sp.detect_objects()
    sp.save_detections(savepath=dtcdir)        

    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(sp.image_bboxes), basewidth=blankimgwidth-100), grid=[1,2])

def predict():
    load_model_in_memory()

    try:
        global sp
    except:
        logger.info("Take a picture first.")

    assert sp.undistorted
    assert sp.segmented, "Segment image first"
    assert sp.detected, "Detect objects first"

    global model, dtcdir, md, expdir, debdir
    md = ModelDetections(dtcdir, img_dim=150, target_classes=insectoptions[:-2])
    
    md.df['full_filename'] = md.df['filename']
    md.df['filename'] = md.df.filename.apply(lambda x: x.split('/')[-1])
    
    md.create_data_generator()
    md.get_predictions(model)

    global df_vals
    df_vals = pd.merge(md.df, sp.yolo_specs, on=['insect_idx'])
    df_vals['user_input'] = 'UNKNOWN'
    df_vals['verified'] = 'UNVERIFIED'
    df_vals.to_csv(f"{debdir}/df_vals_before_overlay.csv")

    disp_img = overlay_yolo(df_vals, sp.image, insectoptions)
    pic_image = Picture(app, image=resize_pil_image(Image.fromarray(disp_img), basewidth=blankimgwidth-100), grid=[1,2])
    df_vals.to_csv(f"{debdir}/df_vals_after_overlay.csv")
    get_interesting_filenames()
    df_vals.to_csv(f"{debdir}/df_vals_after_cleaning.csv")
    
def snap_detect():
    if len(platedate_str.value):
        logger.info(f"Plate date set to: {platedate_str.value}")

        cam = CameraHandler()
        w, h = cam.camera.resolution
        plateloc = plateloc_bt.value
        plateinfo = platenotes_str.value if len(platenotes_str.value) else "NA"
        platedate = platedate_str.value
        platename = f"{plateloc}_{plateinfo}_{platedate}_{w}x{h}.png"
        full_platepath = Path(f"{imgdir}/{platename}")

        cam.capture()
        cam.save(full_platepath)

        global sp
        sp = StickyPlate(full_platepath, caldir)
        sp.undistort(inplace=True)
        # sp.colorcorrect(inplace=True)
        sp.crop_image()
        sp.threshold_image()

        if not plateloc_bt.value.startswith('calibration'):
            sp.detect_objects()
            sp.save_detections(savepath=dtcdir)
            disp_img = sp.image_bboxes
        else:
            disp_img = sp.image

        logger.info("Saved image")

        pic_image = Picture(app, image=resize_pil_image(Image.fromarray(disp_img), basewidth=blankimgwidth-100), grid=[1,2])
        pic_path.value = full_platepath
        del cam
    else:
        app.error(title='Error', text='Is location and date set?')


# ------------- END CAMERA FUNCTIONS -------------
# ------------------------------------------------

#-----------------------------------------------
############ START GUI #########################

def full_run():
    logger.info("Taking picture button pressed")
    check_calib = True # check_calib_done() 
    if check_calib:
        snap_detect()
        update_calib_status()

def do_on_close():
    logger.info("Quit button pressed")
    if yesno("Close", "Are you sure you want to quit?"):
        app.destroy()

def select_location():
    logger.info(f"Selected location: {plateloc_bt.value}")


def select_date():
    logger.info("Date button pressed")
    givendate = app.question("Plate date", "Provide the plate\'s date (e.g. w34 or YYYYMMDD like 20201125).")
    if (givendate.startswith('w') and len(givendate) == 3) or date_validation(givendate):
        if givendate is not None:
            platedate_str.value = givendate
    else:
        app.error(title='Error', text='Date needs to be either in week format: w20 or YYYYMMDD like 20191218.')

def enter_notes():
    logger.info("Notes button pressed")
    givennotes = app.question("Plate notes", "Give some extra location-notes regarding the plate. e.g. 1-60, centroid etc.")
    platenotes_str.value = "NA"
    if len(givennotes) <= 10:
        platenotes_str.value = givennotes
    else:
        platenotes_str.value = "NA"
        app.error(title='Error', text='Length of text provided is too long. Up to 10 characters allowed.')

def date_validation(date_text):
    try:
        datetime.strptime(date_text, '%Y%m%d')
    except:
        logger.info("Incorrect data format, should be YYYYMMDD")
        return False
    return True

#-----------------------------------------------
############ MENU-BAR #########################

def check_session_path(name, created_new=False):
    if created_new:
        user_created_sesspath = f'{default_ses_path}/{name}'
    else:
        user_created_sesspath = name

    default_ses_path_parent = os.path.abspath(os.path.join(default_ses_path, os.pardir))
    print(f"default_ses_path:{default_ses_path}")
    print(f"user_created_sesspath:{user_created_sesspath}")
    print(f"default_ses_path_parent:{default_ses_path_parent}")

    if user_created_sesspath.endswith('images') or user_created_sesspath.endswith('annotations'):
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions" and cannot be \'images\' or \'annotations\'')
        return None
    elif user_created_sesspath.split('/')[-1] == 'sessions':
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions"')
        return None
    elif user_created_sesspath == default_ses_path_parent:
        app.error(title='Error', text='Session path needs to be a subfolder inside "sessions", e.g. sessions/test1/')
        return None
    elif not user_created_sesspath.startswith(default_ses_path_parent):
        app.error(title='Error', text='NOTE: Session path needs to be inside the sessions folder.')
        return None
    else:
        make_dirs([default_ses_path, user_created_sesspath])
        logger.info(f"Created path: {user_created_sesspath}")
        return user_created_sesspath

def change_sess():
    logger.info("Select session folder button pressed")
    name = app.select_folder(folder=default_ses_path)
    selected_sesspath.value = check_session_path(name, created_new=False)

    if selected_sesspath.value is not None:
        global currdir_full
        currdir_full = selected_sesspath.value
        global imgdir, antdir, dtcdir, expdir
        imgdir, antdir, dtcdir, expdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections','exports'])

def create_sess():
    logger.info("Create session button pressed")
    name = app.question("Session folder", "Give a name for the session.")
    selected_sesspath.value = check_session_path(name, created_new=True)

    if selected_sesspath.value is not None:
        global currdir_full
        currdir_full = selected_sesspath.value
        logger.info(f"Session path is set to {currdir_full}.")
        platedate_str.value = ''
        global imgdir, antdir, dtcdir, expdir
        imgdir, antdir, dtcdir, expdir = make_session_dirs(curdir=currdir_full, paths=['images','annotations','detections','exports'])

def open_currdir():
    global currdir_full
    p = subprocess.Popen(["pcmanfm", "%s" % f"{currdir_full}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_imgdir():
    global imgdir
    p = subprocess.Popen(["pcmanfm", "%s" % f"{imgdir}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_dtcdir():
    global dtcdir
    p = subprocess.Popen(["pcmanfm", "%s" % f"{dtcdir}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_logdir():
    global default_log_path
    p = subprocess.Popen(["pcmanfm", "%s" % f"{default_log_path}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_projectdir():
    global default_prj_path
    p = subprocess.Popen(["pcmanfm", "%s" % f"{default_prj_path}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()

def open_expdir():
    global expdir
    p = subprocess.Popen(["pcmanfm", "%s" % f"{expdir}"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.communicate()


#-----------------------------------------------
############ DETECTIONS-VALIDATION #########################

def select_insect():
    global detections_list, val_idx, insect_idx, df_vals
    logger.info(f"Selected insect: {insect_button.value}")
    df_vals.at[insect_idx, 'user_input'] = insect_button.value
    print(df_vals[['insect_idx','prediction','user_input']])

def select_verified():
    global insect_idx, df_vals
    logger.info(f"Selected: {verify_button.value}")
    df_vals.at[insect_idx, 'verified'] = verify_button.value
    print(df_vals[['insect_idx','prediction','user_input','verified']])

def get_interesting_filenames():
    from itertools import chain
    from common import minconf_threshold
    
    global dtcdir, df_vals, debdir

    # Delete insect detections that are not important
    df_vals.to_csv(f"{debdir}/df_vals_debug.csv")

    try:
        # Get a list of filepaths for each row in df_vals that had more than minconf_threshold for a critical insect
        # this list will contain filepaths that will not be deleted 
        list_keep_detections = [df_vals[df_vals[f"{i}"]>=minconf_threshold].full_filename.tolist() for i in critical_insects]
        if not len(list_keep_detections): 
            logger.info("Minimum threshold not crossed for any critical insect")
        list_keep_detections = list(chain.from_iterable(list_keep_detections))
        list_keep_detections.extend(df_vals[df_vals.prediction.isin(critical_insects)].full_filename.tolist()) 
    except:
        # if that list is empty then define the list only based on the top prediction
        list_keep_detections = df_vals[df_vals.prediction.isin(critical_insects)].full_filename.tolist()
    
    # Delete files that are not of interest (no critical insect detections)
    for filepath in df_vals.full_filename.tolist():
        if filepath not in list_keep_detections:
            logger.info(f"Removing {filepath}")
            os.remove(filepath)

    # Removing uninteresting insect detections from dataframe
    df_vals = df_vals[df_vals.full_filename.isin(list_keep_detections)] #.drop(df_vals[~df_vals.prediction.isin(critical_insects)].index, inplace=True)
    # Resetting insect_idx
    df_vals['insect_idx'] = df_vals.index.values
    # Creating a new index for scrolling through insects
    df_vals['scroll_idx'] = df_vals.index.tolist()
    # Resetting index
    df_vals.reset_index(drop=True, inplace=True)
    # Resetting filenames # TODO: this destroys the matching of detection # and insect_idx on prediction
    for i,f in enumerate(natsorted(os.listdir(dtcdir))):
        oldfilename = '_'.join(f.split('_')[:-1])
        old_index = f.split('_')[-1][:-4]
        print(f"f: {f}")
        print(f"oldfilename: {oldfilename}")
        print(f"old_index: {old_index}")
        newfilename = oldfilename + f'_{i}_{old_index}.png'
        print(f"newfilename: {newfilename}")

        shutil.move(f"{dtcdir}/{f}", f"{dtcdir}/{newfilename}")

def open_validation_window():
    assert 'df_vals' in globals(), "You need to first perform model inference to create predictions."

    global dtcdir, detections_list, val_idx, insect_idx, df_vals, critical_insects, confidence_threshold, debdir

    natsorted_detections = natsorted(os.listdir(dtcdir))
    df_vals.filepath = pd.Series(natsorted_detections)
    df_vals.filepath = df_vals.filepath.apply(lambda x: 'detections/'+x)

    val_idx = 0
    print(val_idx)
    # List of all detections' filenames
    detections_list = natsorted([str(fname) for fname in pathlib.Path(dtcdir).glob('**/*.png')])
    assert len(detections_list), "No critical insects detected. Nothing to verify."
    print(detections_list)
    # Display image of the validations window
    print(detections_list[val_idx])
    val_img.image = detections_list[val_idx]
    # Insect index taken from the filename
    insect_idx = df_vals.loc[val_idx].insect_idx #int(detections_list[val_idx].split('_')[-1][:-4])
    # All insect probability scores taken from df_vals
    pred_dict = df_vals[insectoptions[:-2]].loc[val_idx].apply(lambda x: int(round(x,0))).sort_values(ascending=False).to_dict()
    pred_str = str(pred_dict)
    pred_str = pred_str[1:-1].replace('\'','').replace(': ',':').replace(',','%')+'%'
    # Displaying the detection number (bounding box number)
    detectioninfo_str.value = f'DETECTION #{insect_idx}'
    # Displaying all probabilities per insect class
    predictioninfo_str.value = f'{pred_str}'
    # Setting and displaying the drop-down insect class button to the chosen value
    insect_button.value = df_vals.loc[val_idx].user_input
    # Drop-down verification button (VERIFIED/UNVERIFIED)
    verify_button.value = df_vals.loc[val_idx].verified
    # Setting a class name for insects with high prediction confidence value
    if not df_vals.loc[val_idx].uncertain and verify_button.value == "UNVERIFIED":
        insect_button.value = df_vals.loc[val_idx].prediction

    df_vals.to_csv(f"{debdir}/df_vals_after_valwindow.csv")

    val_window.show()

def next_val():
    """ Similar to open_validation_window """
    global dtcdir, detections_list, val_idx, insect_idx, df_vals, insectoptions, confidence_threshold

    if val_idx < len(detections_list)-1:
        val_idx+=1
        val_img.image = detections_list[val_idx]
    else:
        val_idx = 0
        val_img.image = detections_list[val_idx]

    insect_idx = df_vals.loc[val_idx].insect_idx #insect_idx = int(detections_list[val_idx].split('_')[-1][:-4])
    pred_dict = df_vals[insectoptions[:-2]].loc[val_idx].apply(lambda x: round(x,1)).sort_values(ascending=False).to_dict()
    pred_str = str(pred_dict)
    pred_str = pred_str[1:-1].replace('\'','').replace(': ',':').replace(',','%')+'%'
    detectioninfo_str.value = f'DETECTION #{insect_idx}'
    predictioninfo_str.value = f'{pred_str}'
    insect_button.value = df_vals.loc[val_idx].user_input
    verify_button.value = df_vals.loc[val_idx].verified
    if not df_vals.loc[val_idx].uncertain and verify_button.value == "UNVERIFIED":
        insect_button.value = df_vals.loc[val_idx].prediction

def prev_val():
    """ Similar to open_validation_window """
    global dtcdir, detections_list, val_idx, insect_idx, df_vals, confidence_threshold

    if val_idx > 0:
        val_idx-=1
        val_img.image = detections_list[val_idx]
    else:
        val_idx = len(detections_list)-1
        val_img.image = detections_list[val_idx]

    insect_idx = df_vals.loc[val_idx].insect_idx #insect_idx = int(detections_list[val_idx].split('_')[-1][:-4])
    pred_dict = df_vals[insectoptions[:-2]].loc[val_idx].apply(lambda x: round(x,1)).sort_values(ascending=False).to_dict()
    pred_str = str(pred_dict)
    pred_str = pred_str[1:-1].replace('\'','').replace(': ',':').replace(',','%')+'%'
    detectioninfo_str.value = f'DETECTION #{insect_idx}'
    predictioninfo_str.value = f'{pred_str}'
    insect_button.value = df_vals.loc[val_idx].user_input
    verify_button.value = df_vals.loc[val_idx].verified    
    if not df_vals.loc[val_idx].uncertain and verify_button.value == "UNVERIFIED":
        insect_button.value = df_vals.loc[val_idx].prediction

def close_validation_window():
    val_window.hide()
    if yesno("Close", "Are you sure you want to exit without saving?"):
        val_window.hide()
    else:
        val_window.show()

def save_and_reset():
    global df_vals, currdir_full, critical_insects, dtcdir, expdir

    # Create directories per class name in exports folder
    export_folders = np.unique(df_vals.prediction.tolist() + df_vals.user_input.tolist()+['UNKNOWN']).tolist()
    for p in export_folders:
        path = f"{expdir}/{p}/"
        if not os.path.exists(path):
            os.makedirs(path)
    # Create the filenames of the exports
    new_paths = []
    for i, row in df_vals.iterrows():
       new_paths.append(f"{expdir}/{row.user_input}/{row.full_filename.split('/')[-1][:-4]}_{row.prediction}_{row.user_input}_{row.verified}.png")
    # Moving detections to the exports folder
    for i in range(len(new_paths)):
        oldpath = f"{dtcdir}/{df_vals.filepath.tolist()[i].replace('detections/','')}"
        newpath = new_paths[i]

        shutil.move(oldpath, newpath, copy_function=shutil.copy2)


    # Exporting results in a csv
    assert df_vals.pname.unique().shape[0] == 1, "More sticky plates in dataframe?"
    platename = df_vals.pname.unique()[0]
    df_vals.to_csv(f"{currdir_full}/{platename}_dataframe_all_results.csv")
    df_vals.top_class.value_counts().to_csv(f"{currdir_full}/{platename}_dataframe_pests_model_results.csv")
    df_vals.user_input.value_counts().to_csv(f"{currdir_full}/{platename}_dataframe_pests_user_results.csv")

    
    # Creating some summary plots to export    
    if not os.path.isdir(f"{currdir_full}/plots/"):
        os.makedirs(f"{currdir_full}/plots/")

    plt.figure()
    df_vals.top_class.value_counts().plot(kind='bar')
    plt.title(platename)
    plt.savefig(f"{currdir_full}/plots/{platename}_model_prediction_summary.png", bbox_inches='tight')
    plt.close()
    
    plt.figure()
    df_vals.user_input.value_counts().plot(kind='bar')
    plt.title(platename)
    plt.savefig(f"{currdir_full}/plots/{platename}_user_input_summary.png", bbox_inches='tight')
    plt.close()

    
    memory_reset()
    val_window.hide()

#-----------------------------------------------
############ GENERAL #########################

def memory_reset():
    global sp, df_vals, detections_list, md
    del sp, df_vals, detections_list, md
    pic_image = Picture(app, image=Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0)), grid=[1,2])

def update_calib_status(nr_calib_plates=False):
    global currdir_full

    chessboard_imgs_in_currdir = glob.glob(f'{caldir}/calibration_chessboard*.jpg')
    color_img_in_currdir = glob.glob(f'{caldir}/calibration_color*.jpg')
    calib_chess_st.value = f"Calib chess: {len(chessboard_imgs_in_currdir)}"
    calib_color_st.value = f"Calib color: {len(color_img_in_currdir)}"
    logger.info(f"Chessboard images: {len(chessboard_imgs_in_currdir)}")
    logger.info(f"Colorplate image: {len(color_img_in_currdir)}")
    selected_sesspath.value = currdir_full

    if nr_calib_plates:
        return len(chessboard_imgs_in_currdir), len(color_img_in_currdir)

def check_calib_done():
    if plateloc_bt.value not in ["other", "calibration_chessboard", "calibration_color"]:
        nr_chess_imgs, nr_color_imgs = update_calib_status(nr_calib_plates=True)
        if nr_chess_imgs < 10 or nr_color_imgs < 1:
            app.error(title='Error', text=f'Please perform calibration first. Minimum of 10 chessboard images and one Color plate image. \
                                            Found {nr_chess_imgs} chessboard images and {nr_color_imgs} color plate(s).')
            return False
        else:
            calib_chess_st.value = f"Chessboard images: OK"
            calib_color_st.value = f"Colorplate images: OK"
    return True

def load_model_in_memory():
    logger.info("Loading Insect-Model in memory..")
    global model, md, dtcdir, modelname
    assert len(os.listdir(dtcdir)), "No insects detected (yet).."
    md = ModelDetections(dtcdir, img_dim=150, target_classes=insectoptions[:-2])
    model = InsectModel(img_dim=md.img_dim, nb_classes=len(md.target_classes), modelname=modelname).load()

def get_stats():
    logger.debug("Attempting to get cpu temperature")
    global cpu_temperature
    cpu_temperature = get_cpu_temperature()
    cputemp_st.value = f"CPUtemp: {cpu_temperature}"



# ------------- STOP GUI -------------
# ------------------------------------

# ------------------------------------
# ------------- MAIN -----------------
if __name__=="__main__":

    app = App(title="Photobox v0.0.3", layout="grid", width=appwidth, height=appheight, bg = background)

    # MENU
    menubar             = MenuBar(app, 
                                toplevel=["File","Open"],
                                options=[
                                    [ 
                                        ["Change current session..", change_sess], 
                                        ["New session..", create_sess] 
                                    ],
                                    [ 
                                        ["Open session directory..", open_currdir], 
                                        ["Open image directory..", open_imgdir], 
                                        ["Open detections directory..", open_dtcdir],
                                        ["Open exports directory..", open_expdir],
                                        ["Open logs directory..", open_logdir] ,  
                                        ["Open project directory..(to change locations, insects)", open_projectdir], 
                                    ], ]                        
                                    )


    # MAIN INTERFACE
    sess                = Text(app, grid=[0,0], align='right')
    sess.value          = 'Current session:'
    selected_sesspath   = Text(app, grid=[1,0], align='left')
    selected_sesspath.after(1, create_sess)
    plateloc_bt         = Combo(app, 
                                        options=platelocations, 
                                        command=select_location, 
                                        selected="LOCATION", 
                                        grid=[0,3],
                                        align='right')    
    platenotes_bt       = PushButton(app, text='INFO', command=enter_notes, grid=[0,4], align='right')
    platenotes_str      = Text(app, grid=[1,4], align='left')
    platedate_bt        = PushButton(app, text='DATE', command=select_date, grid=[0,5], align='right')
    platedate_str       = Text(app, grid=[1,5], align='left')
    pic_image           = Picture(app, image=Image.new('RGB', (blankimgwidth, blankimgheight), (0,0,0)), grid=[1,2])
    last_img            = Text(app, grid=[0,6], align='right')
    last_img.value      = "LAST IMAGE:"
    pic_path            = Text(app, grid=[1,6], align='left')


    # STATS BOX
    stats_box           = Box(app, height='fill', align='right', grid=[2,4])
    calib_chess_st      = Text(stats_box, align='top')
    calib_color_st      = Text(stats_box, align='top')
    cputemp_st          = Text(stats_box, align='top')
    cputemp_st.value    = f"CPUtemp: {cpu_temperature}"
    cputemp_st.repeat(5000, get_stats)
    calib_color_st.after(2, update_calib_status)


    # MAIN IMG PROCESSING BOX
    imgproc_box         = Box(app, height="fill", align="right", grid=[0,2])

    space1              = Drawing(imgproc_box, align='top')
    space1.rectangle(10,10,60,60, color=background)
    snap_button         = PushButton(imgproc_box, command=snap, text="CAPTURE", align='top', image='icons/camera.png')
    calib_button        = PushButton(imgproc_box, command=calibrate, text="CALIBRATE", align='top', image='icons/calibrate.png')
    crop_button         = PushButton(imgproc_box, command=crop, text="CROP", align='top', image='icons/crop.png')
    segment_button      = PushButton(imgproc_box, command=segment, text="SEGMENT", align='top', image='icons/segment.png')
    detect_button       = PushButton(imgproc_box, command=detect, text="DETECT", align='top', image='icons/detect.png')
    predict_button      = PushButton(imgproc_box, command=predict, text="PREDICT", align='top', image='icons/predict.png')
    space2              = Drawing(imgproc_box, align='top')
    space2.rectangle(10,10,60,60, color=background)


    # FULLRUN BOX
    fullrun_box         = Box(app, width="fill", align="right", grid=[1,3])
    
    openval_button      = PushButton(fullrun_box, command=open_validation_window, text="3. VALIDATE PREDICTIONS", align='right', image='icons/validate.png')
    predict_bt          = PushButton(fullrun_box, command=predict, text="2. MODEL INFERENCE", align='right', image='icons/model.png')
    fullrun_bt          = PushButton(fullrun_box, command=full_run, text="1. PROCESS IMAGE", align='right', image='icons/process.png')


    # VALIDATION WINDOW
    val_window          = Window(app, visible=False, bg='white', width=650, height=650, title="VALIDATE DETECTIONS")

    savexit_button      = PushButton(val_window, text="SAVE & RESET", command=save_and_reset, align='top')
    closeval_button     = PushButton(val_window, text="EXIT", command=close_validation_window, align='top')
    val_img             = Picture(val_window, image=Image.new('RGB', (blankimgwidth//2, blankimgheight//2), (0,0,0)), align='top')
    insect_button       = Combo(val_window, options=insectoptions, 
                                    command=select_insect,
                                    selected="UNKNOWN", 
                                    align='top')
    verify_button       = Combo(val_window, options=['UNVERIFIED','VERIFIED'], 
                                    command=select_verified,    
                                    selected="UNVERIFIED", 
                                    align='top')
    next_val_button     = PushButton(val_window, command=next_val, text="NEXT", align='top', image='icons/next1.png')
    prev_val_button     = PushButton(val_window, command=prev_val, text="PREVIOUS", align='top', image='icons/prev.png')
    detectioninfo_str   = Text(val_window, align='top')
    detectioninfo_str.value = 'DETECTION #?'
    predictioninfo_str  = Text(val_window, align='top')
    predictioninfo_str.value = ''

    app.when_closed = do_on_close
    app.display()
