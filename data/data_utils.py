import sys
import os
import numpy as np



type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
class2type = {type2class[t]:t for t in type2class}
type2onehotclass={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}
type_mean_size = {'bathtub': np.array([0.765840,1.398258,0.472728]),
                  'bed': np.array([2.114256,1.620300,0.927272]),
                  'bookshelf': np.array([0.404671,1.071108,1.688889]),
                  'chair': np.array([0.591958,0.552978,0.827272]),
                  'desk': np.array([0.695190,1.346299,0.736364]),
                  'dresser': np.array([0.528526,1.002642,1.172878]),
                  'night_stand': np.array([0.500618,0.632163,0.683424]),
                  'sofa': np.array([0.923508,1.867419,0.845495]),
                  'table': np.array([0.791118,1.279516,0.718182]),
                  'toilet': np.array([0.699104,0.454178,0.756250])}
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 10
NUM_CLASS = 10

def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc[:,[0,2]] = np.dot(pc[:,[0,2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
       
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
    return class_id, residual_angle

def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle>np.pi:
        angle = angle - 2*np.pi
    return angle
        
def size2class(size, type_name):
    ''' Convert 3D box size (l,w,h) to size class and size residual '''
    size_class = type2class[type_name]
    size_residual = size - type_mean_size[type_name]
    return size_class, size_residual

def class2size(pred_cls, residual):
    ''' Inverse function to size2class '''
    mean_size = type_mean_size[class2type[pred_cls]]
    return mean_size + residual

# def get_3d_box(box_size, heading_angle, center):
#     ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
#         output (8,3) array for 3D box cornders
#         Similar to utils/compute_orientation_3d
#     '''
#     R = roty(heading_angle)
#     l,w,h = box_size
#     x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
#     y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
#     z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
#     corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
#     corners_3d[0,:] = corners_3d[0,:] + center[0];
#     corners_3d[1,:] = corners_3d[1,:] + center[1];
#     corners_3d[2,:] = corners_3d[2,:] + center[2];
#     corners_3d = np.transpose(corners_3d)
#     return corners_3d

# def compute_box3d_iou(center_pred, heading_logits, heading_residuals, size_logits, size_residuals, center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
#     ''' Used for confidence score supervision..
#     Inputs:
#         center_pred: (B,3)
#         heading_logits: (B,NUM_HEADING_BIN)
#         heading_residuals: (B,NUM_HEADING_BIN)
#         size_logits: (B,NUM_SIZE_CLUSTER)
#         size_residuals: (B,NUM_SIZE_CLUSTER,3)
#         center_label: (B,3)
#         heading_class_label: (B,)
#         heading_residual_label: (B,)
#         size_class_label: (B,)
#         size_residual_label: (B,3)
#     Output:
#         iou2ds: (B,) birdeye view oriented 2d box ious
#         iou3ds: (B,) 3d box ious
#     '''
#     batch_size = heading_logits.shape[0]
#     heading_class = np.argmax(heading_logits, 1) # B
#     heading_residual = np.array([heading_residuals[i,heading_class[i]] for i in range(batch_size)]) # B,
#     size_class = np.argmax(size_logits, 1) # B
#     size_residual = np.vstack([size_residuals[i,size_class[i],:] for i in range(batch_size)])

#     iou2d_list = [] 
#     iou3d_list = [] 
#     for i in range(batch_size):
#         heading_angle = class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
#         box_size = class2size(size_class[i], size_residual[i])
#         corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

#         heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
#         box_size_label = class2size(size_class_label[i], size_residual_label[i])
#         corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])

#         iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label) 
#         iou3d_list.append(iou_3d)
#         iou2d_list.append(iou_2d)
#     return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)

# def compare_with_anchor_boxes(center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
#     ''' Compute IoUs between GT box and anchor boxes.
#         Compute heading,size,center regression from anchor boxes to GT box: NHxNS of them in the order of
#             heading0: size0,size1,...
#             heading1: size0,size1,...
#             ...
#     Inputs:
#         center_label: (B,3) -- assume this center is already close to (0,0,0) e.g. subtracted stage1_center
#         heading_class_label: (B,)
#         heading_residual_label: (B,)
#         size_class_label: (B,)
#         size_residual_label: (B,3)
#     Output:
#         iou2ds: (B,K) where K = NH*NS
#         iou3ds: (B,K) 
#         center_residuals: (B,K,3)
#         heading_residuals: (B,K)
#         size_residuals: (B,K,3)
#     '''
#     B = len(heading_class_label)
#     K = NUM_HEADING_BIN*NUM_SIZE_CLUSTER
#     iou3ds = np.zeros((B,K), dtype=np.float32)
#     iou2ds = np.zeros((B,K), dtype=np.float32)
#     center_residuals = np.zeros((B,K,3), dtype=np.float32)
#     heading_residuals = np.zeros((B,K), dtype=np.float32)
#     size_residuals = np.zeros((B,K,3), dtype=np.float32)
 
#     corners_3d_anchor_list = []
#     heading_anchor_list = []
#     box_anchor_list = []
#     for j in range(NUM_HEADING_BIN):
#        for k in range(NUM_SIZE_CLUSTER):
#            heading_angle = class2angle(j,0,NUM_HEADING_BIN)
#            box_size = class2size(k,np.zeros((3,)))
#            corners_3d_anchor = get_3d_box(box_size, heading_angle, np.zeros((3,)))
#            corners_3d_anchor_list.append(corners_3d_anchor)
#            heading_anchor_list.append(heading_angle)
#            box_anchor_list.append(box_size)

#     for i in range(B):
#         heading_angle_label = class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
#         box_size_label = class2size(size_class_label[i], size_residual_label[i])
#         corners_3d_label = get_3d_box(box_size_label, heading_angle_label, center_label[i])
#         for j in range(K):
#             iou_3d, iou_2d = box3d_iou(corners_3d_anchor_list[j], corners_3d_label) 
#             iou3ds[i,j] = iou_3d
#             iou2ds[i,j] = iou_2d
#             center_residuals[i,j,:] = center_label[i]
#             heading_residuals[i,j] = heading_angle_label - heading_anchor_list[j]
#             size_residuals[i,j,:] = box_size_label - box_anchor_list[j]

#     return iou2ds, iou3ds, center_residuals, heading_residuals, size_residuals