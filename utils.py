from shutil import copy2
from tqdm import tqdm
import numpy as np

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

    for _, row in tqdm(specifications.iterrows(), desc="Saving detections.."):
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

def overlay_yolo(specifications, plate_img, class_selection):
    import cv2

    H,W,_ = plate_img.shape
    
    for _, row in tqdm(specifications.iterrows(), "Overlaying bboxes with predictions.."):
        if row.prediction in class_selection:

            left  = int((row.yolo_x-row.yolo_width/2.)*W)
            right = int((row.yolo_x+row.yolo_width/2.)*W)
            top   = int((row.yolo_y-row.yolo_height/2.)*H)
            bot   = int((row.yolo_y+row.yolo_height/2.)*H)

            if(left < 0): left = 0;
            if(right > W-1): right = W-1;
            if(top < 0): top = 0;
            if(bot > H-1): bot = H-1;

            if row.uncertain:
                cv2.rectangle(plate_img, (left, top), (right, bot), (255, 0, 0), 2)
                cv2.putText(plate_img, f"{row.insect_id},{row.prediction}.{row.top_prob/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (255,0,0), 2)
            else:
                cv2.rectangle(plate_img, (left, top), (right, bot), (255, 255, 0), 2)
                cv2.putText(plate_img, f"{row.insect_id},{row.prediction}.{row.top_prob/100:.0%}", (left-10, top-20), cv2.FONT_HERSHEY_COMPLEX, 1., (0,255,0), 2)
    return plate_img