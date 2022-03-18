import numpy as np
import pandas as pd
import os
import cv2
from utils import copy_files, non_max_suppression_fast, save_insect_crops, overlay_yolo
from PIL import Image
import glob
import time
from tqdm import tqdm
from matplotlib import cm
import logging
import warnings
from common import crop_pxls_top, crop_pxls_bot, crop_pxls_left, crop_pxls_right, nms_threshold, bnw_threshold, min_obj_area, max_obj_area
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)

class StickyPlate(object):

    def __init__(self, path, chessboard_dir):
        self.path = str(path)
        self.pname = self.path.split('/')[-1][:-4]
        self.image = np.array(Image.open(self.path))
        self.H, self.W = self.image.shape[:2]
        self.chessboard_dir = chessboard_dir
        self.cropped = False

    def undistort(self, findpoints=False, inplace=True, verbose=False):
        logger.info("Undistorting..")

        if findpoints:
            # termination criteria
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((7*7,3), np.float32)
            objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * 10

            # Arrays to store object points and image points from all the images.
            objpoints = [] # 3d point in real world space
            imgpoints = [] # 2d points in image plane.

            chessboard_images = glob.glob(f'{self.chessboard_dir}/*.jpg')
            assert len(chessboard_images) > 5, "Too few chessboard images for this session"
            assert len(chessboard_images) < 18, f"Too many chessboard images ({len(chessboard_images)}) for this session. Maybe not all of them are chessboard images?"


            """ FINDING CHESSBOARD POINTS """

            a, b = 7,7 # chessboard dims
    
            successes = 0
            for fpath in tqdm(chessboard_images, desc='Calibrating using the chessboard images..'):
                fname = fpath.split('/')[-1][:-4]

                if verbose:
                    print(f'\nPROCESSING : {fname}')

                if "color" in fname or "empty" in fname:
                    print("Color image detected. Please use ONLY chessboard images. Ignoring this one.")
                    continue

                assert "calibration" in fname, "Check that you put chessboard images only in this folder."

                t = time.time()
                img = cv2.imread(fpath)
                if img is None:
                    raise ValueError("Image not read from opencv")
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray,9,55,55)
                gray = cv2.medianBlur(gray, 5)
                gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1] # threshold = 127  

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (a,b),None)

                # If found, add object points, image points (after refining them)
                if ret == True:
                    successes+=1
                    if verbose:
                        print("SUCCESS: Found points successfully! Adding object points.")
                    
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    imgpoints.append(corners2)

                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
                    if verbose:
                        cv2.imshow('img',cv2.resize(img, (828, 746)))
                        cv2.waitKey(100)
                else:
                    if verbose:
                        print("FAILED: Was not able to find the object points.")
                if verbose:
                    print(f'Elapsed time for {fname}: {time.time() - t} seconds')
            
            cv2.destroyAllWindows()

            if successes < 7:
                raise KeyError("Not enough chessboard images were processed succesfully..")

            """ CAMERA CALIBRATION """

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            np.savez(f'{self.chessboard_dir}/calib_params.npz', ret=ret, mtx=mtx, dist=dist,
                                                    rvecs=rvecs, tvecs=tvecs)

        """ UNDISTORTION """

        data = np.load(f'{self.chessboard_dir}/calib_params.npz')

        fpath = self.path
        fname = fpath.split('/')[-1][:-4]
        if verbose:
            print(f'\nUNDISTORTING : {fname}')

        ret, mtx, dist, rvecs, tvecs = data['ret'], data['mtx'], data['dist'], data['rvecs'], data['tvecs']

        img = self.image
        
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        self.undistorted = True

        if inplace:
            self.image = dst
        else:
            self.undistorted_img = dst

    def colorcorrect(self, inplace=True, plot=False):
        logger.info("Color correction..")
        import colour
        import matplotlib.pyplot as plt
        from colour_checker_detection import (colour_checkers_coordinates_segmentation,
                                            detect_colour_checkers_segmentation)
        from colour_checker_detection.detection.segmentation import adjust_image

        colour_checker_image_path = [f"{str(self.chessboard_dir)}/pb_colorcalib3.png"] 
        colour_checker_imgs = [colour.cctf_decoding(colour.io.read_image(path)) for path in colour_checker_image_path]

        # Detection
        SWATCHES = []
        for image in colour_checker_imgs:
            for swatches, _, _ in detect_colour_checkers_segmentation( image, additional_data=True):
                SWATCHES.append(swatches)

        # Colour Fitting
        D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        reference_colour_checker = colour.CCS_COLOURCHECKERS['ColorChecker 2005']
        reference_swatches = colour.XYZ_to_RGB(
                colour.xyY_to_XYZ(list(reference_colour_checker.data.values())),
                reference_colour_checker.illuminant, D65,
                colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB)

        # Custom image colour correction
        calibrated = colour.cctf_encoding(colour.colour_correction(colour.cctf_decoding(self.image/255.), SWATCHES[0], reference_swatches))
        if plot:
            colour.plotting.plot_image(calibrated)

        if inplace:
            self.image = (calibrated * 255).astype(np.uint8)
        else:
            self.calibrated = (calibrated * 255).astype(np.uint8)

    def crop_image(self, crop_pxls_top=crop_pxls_top, crop_pxls_bot=crop_pxls_bot, crop_pxls_left=crop_pxls_left, crop_pxls_right=crop_pxls_right):
        logger.info("Cropping..")

        print(f"Original image shape: {self.image.shape}")
        height,width = self.image.shape[:2]
        self.image = self.image[crop_pxls_top:height-crop_pxls_bot,
                                crop_pxls_left:width-crop_pxls_right]
        self.H, self.W = self.image.shape[:2]

        print(f"New image shape: {self.image.shape}")

    def threshold_image(self, bnw_threshold=bnw_threshold):
        logger.info("Thresholding..")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.blurred = cv2.medianBlur(self.gray,5)

        ret,th1 = cv2.threshold(self.blurred,bnw_threshold,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        th1 = cv2.dilate(th1, kernel, iterations=1)

        self.thresholded = th1
        self.pil_thresholded = Image.fromarray(th1)
        self.segmented = True

    def detect_objects(self, min_obj_area=min_obj_area, max_obj_area=max_obj_area, nms_threshold=nms_threshold, insect_img_dim=150):
        logger.info("Detecting objects..")

        assert hasattr(self, 'thresholded'), "Threshold the image before detecting objects."
        contours = cv2.findContours(self.thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        coordinates = []
        for c in contours:
            rect = cv2.boundingRect(c)
            if rect[2] < min_obj_area or rect[3] < min_obj_area: continue
            if cv2.contourArea(c) > max_obj_area: continue
            x,y,w,h = rect
            
            coordinates.append((x,y,x+w, y+h))

        self.bbox_coordinates = np.array(coordinates)
        assert len(coordinates), "No objects detected!"
        pick, idxs = non_max_suppression_fast(self.bbox_coordinates, nms_threshold)
        self.image_bboxes = self.image.copy()
        for i, (startX, startY, endX, endY) in enumerate(pick):
            cv2.rectangle(self.image_bboxes, (startX, startY), (endX, endY), (0, 255, 0), 2)

        yolo_x,yolo_y,yolo_w,yolo_h = [],[],[],[]
        for i in range(len(self.bbox_coordinates)):
            x1,y1,x2,y2 = self.bbox_coordinates[i]
            w,h = np.abs(x2-x1), np.abs(y1-y2)

            yolo_x.append(np.abs(x2 - np.abs(x1 - x2) /2) / self.W)
            yolo_y.append(np.abs(y2 - np.abs(y1 - y2) /2) / self.H)
            yolo_w.append(insect_img_dim/self.W)
            yolo_h.append(insect_img_dim/self.H)
            
        yolo_specs = pd.DataFrame({"yolo_x":yolo_x, "yolo_y":yolo_y, "yolo_width":yolo_w, "yolo_height":yolo_h})
        yolo_specs['pname'] = self.pname
        yolo_specs['insect_idx'] = yolo_specs.index.values.astype(str)    
        self.yolo_specs = yolo_specs
        self.detected = True

    def save_detections(self, savepath="./detections/"):
        logger.info("Saving deteections..")

        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_insect_crops(self.yolo_specs, savepath, self.image)

def resize_pil_image(img, basewidth=300):
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    return img
