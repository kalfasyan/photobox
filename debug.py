from camera import CameraHandler
from detections import *
from stickyplate import *

cam = CameraHandler()
cam.capture()
cam.save('./test.png')

sp = StickyPlate('./test.png', './calib/')
sp.undistort()
sp.threshold_image()
sp.detect_objects()
sp.save_detections(savepath='./detections/')

md = ModelDetections('./detections/')
md.create_data_generator()
model = InsectModel('insectmodel.h5', md.img_dim, len(md.target_classes)).load()
md.get_predictions(model, sp)
