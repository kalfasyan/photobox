import numpy as np
import cv2
import glob
import time
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pi", required=True,
	help="Raspberry pi index")
args = vars(ap.parse_args())


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2) * 10

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

pi_idx = args["pi"]
images = glob.glob(f'p{pi_idx}/*.jpg')


for fpath in images:
    fname = fpath.split('/')[-1][:-4]
    print(f'PROCESSING : {fname}')
    #break
    t = time.time()
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = cv2.resize(gray, (828, 746))
    gray = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1] # threshold = 127  
    cv2.imwrite(f'p{pi_idx}/output_blacknwhite/{fname}_black_nwhite.jpg',gray)  
    #break
    
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Found points successfully! Adding object points.")
        
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,7), corners2,ret)
        cv2.imshow('img',cv2.resize(img, (828, 746)))
        cv2.waitKey(1500)
        cv2.imwrite(f'p{pi_idx}/output_chessboards/{fname}_chessboard.jpg',img)
    print(f'ELAPSED TIME FOR {fname}: {time.time() - t}')
    #break
cv2.destroyAllWindows()

## CAMERA CALIBRATION

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

np.savez(f'p{pi_idx}_calib_params.npz', ret=ret, mtx=mtx, dist=dist,
                                        rvecs=rvecs, tvecs=tvecs)

