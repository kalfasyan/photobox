from shutil import copy2
from tqdm import tqdm
import numpy as np
from common import default_ses_path, minconf_threshold, confidence_threshold
import pathlib
import os

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

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), idxs

def copy_files(filelist, destination):
    for f in tqdm(filelist, total=len(filelist), desc="Copying files.."):
        copy2(f, destination)

def save_insect_crops(specifications, path_crops, plate_img):
    import cv2
    import os
    from PIL import Image

    os.system(f"rm -rf {path_crops}/*")
    
    H,W,_ = plate_img.shape

    for _, row in tqdm(specifications.iterrows(), total=specifications.shape[0], desc="Saving detections.."):
        left  = int((row.yolo_x-row.yolo_width/2.)*W)
        right = int((row.yolo_x+row.yolo_width/2.)*W)
        top   = int((row.yolo_y-row.yolo_height/2.)*H)
        bot   = int((row.yolo_y+row.yolo_height/2.)*H)

        if(left < 0): left = 0;
        if(right > W-1): right = W-1;
        if(top < 0): top = 0;
        if(bot > H-1): bot = H-1;

        # print(f"left: {left}, right: {right}, top: {top}, bot: {bot}")
        crop = plate_img[top:bot, left:right]

        savepath = f"{path_crops}/"
        if not os.path.isdir(savepath):
            os.makedirs(savepath)

        crop = Image.fromarray(crop)
        crop.save(f"{savepath}/{row.pname}_{row.name}.png")

def overlay_yolo(specifications, plate_img, class_selection, confidence_threshold=confidence_threshold, top_classes=['wmv','wswl']):
    import cv2

    assert all(i in class_selection for i in top_classes), "Top class does not belong in class selection."
    # assert confidence_threshold > 20, "Threshold too low. Set it above 20."

    H,W,_ = plate_img.shape
    print(specifications)
    for _, row in tqdm(specifications.iterrows(), total=specifications.shape[0], desc="Overlaying bboxes with predictions.."):
        if row.prediction in class_selection:

            left  = int((row.yolo_x-row.yolo_width/2.)*W)
            right = int((row.yolo_x+row.yolo_width/2.)*W)
            top   = int((row.yolo_y-row.yolo_height/2.)*H)
            bot   = int((row.yolo_y+row.yolo_height/2.)*H)

            if(left < 0): left = 0;
            if(right > W-1): right = W-1;
            if(top < 0): top = 0;
            if(bot > H-1): bot = H-1;

            if row.prediction in top_classes or (row[top_classes] >= minconf_threshold).any():
                #print(f"row.prediction:{row.prediction} in [top_classes]: {top_classes}")
                # if any of the top classes has larger prob than minconf_threshold AND NOT any of top classes has larger than confidence_threshold
                #print(row[top_classes])
                if (row[top_classes] >=minconf_threshold).any() and not (row[top_classes] >= confidence_threshold).any():
                    max_of_top_classes = row[top_classes].index[row[top_classes].values.argmax()]
                    #print(f"max of two classes: {max_of_top_classes}")
                    if max_of_top_classes == 'wmv' and row.wmv > confidence_threshold/2:
                        cv2.rectangle(plate_img, (left, top), (right, bot), (0, 255, 0), 2)
                        cv2.putText(plate_img, f"{row.insect_idx},{max_of_top_classes}.{row[max_of_top_classes]/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (0,255,0), 2)
                        continue                        
                    cv2.rectangle(plate_img, (left, top), (right, bot), (255, 0, 0), 2)
                    cv2.putText(plate_img, f"{row.insect_idx},{max_of_top_classes}.{row[max_of_top_classes]/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (255,0,0), 2)
                else:
                    cv2.rectangle(plate_img, (left, top), (right, bot), (255, 255, 0), 2)
                    cv2.putText(plate_img, f"{row.insect_idx},{row.prediction}.{row.top_prob/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (0,255,0), 2)
            else:
                cv2.rectangle(plate_img, (left, top), (right, bot), (0, 0, 255), 2)
                cv2.putText(plate_img, f"{row.insect_idx},{row.prediction}.{row.top_prob/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (0, 0, 255), 2)
            
    return plate_img

def get_cpu_temperature():
    return ''
    # import subprocess
    # import re
    # tmp = subprocess.check_output(["vcgencmd measure_temp"], shell=True)
    # cputemp = tmp.decode('ascii')
    # cputemp = re.findall("\d+", cputemp)
    # cputemp = '.'.join(cputemp)
    # return cputemp
