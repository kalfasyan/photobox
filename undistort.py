import cv2
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--pi", required=True,
	help="Raspberry pi index")
ap.add_argument("-i", "--idx", required=True,
	help="Image index")
args = vars(ap.parse_args())

pi_idx = args["pi"]
img_idx = args["idx"]

data = np.load(f'p{pi_idx}_calib_params.npz')

ret, mtx, dist, rvecs, tvecs = data['ret'], data['mtx'], data['dist'], data['rvecs'], data['tvecs']

image_undistort = f'RPi4_p{pi_idx}_i{img_idx}'
img = cv2.imread(f'p{pi_idx}/{image_undistort}.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(f'{image_undistort}_CALIBRESULT.png',dst)
