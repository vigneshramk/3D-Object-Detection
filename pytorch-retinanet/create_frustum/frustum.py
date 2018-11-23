import os
import sys
import numpy as np
import sys
import utils
import scipy

# Get the directories
root_dir = '/home/kvr/Documents/Projects/3D-Object-Detection/2d_data'
image_dir = os.path.join(root_dir, 'image')
label_dir = os.path.join(root_dir, 'label_dimension')
depth_dir = os.path.join(root_dir, 'depth2')
calib_dir = os.path.join(root_dir, 'calib')


def get_image(idx):
    img_filename = os.path.join(image_dir, '%06d.jpg'%(idx))
    return utils.load_image(img_filename)

def get_depth(idx): 
    depth_filename = os.path.join(depth_dir, '%06d.mat'%(idx))
    return utils.load_depth_points(depth_filename)

def get_calibration(idx):
    calib_filename = os.path.join(calib_dir, '%06d.txt'%(idx))
    return utils.SUNRGBD_Calibration(calib_filename)

def get_label_objects(idx):
    label_filename = os.path.join(label_dir, '%06d.txt'%(idx))
    return utils.read_sunrgbd_label(label_filename)


for data_idx in range(1,10):

	# Get calib, objects, image
	calib = get_calibration(data_idx)
	objects = sunrgbd.get_label_objects(data_idx)
	img = get_image(data_idx)

	# Load and convert depth data from mat files
	pc_upright_depth = get_depth(data_idx)
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:,0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:,0:3])
    pc_upright_camera[:,3:] = pc_upright_depth[:,3:]
    





