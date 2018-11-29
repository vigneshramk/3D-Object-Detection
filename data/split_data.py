from loader import SUNRGBD
import numpy as np
import cv2
import os
import pickle
import gzip

def save_zipped_pickle(obj, filename, protocol=-1):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def split_dataset():
    train_factor = 0.7
    loader = SUNRGBD()
    id_list,box2d_list,box3d_list,input_list,label_list,type_list,heading_list,size_list,frustum_angle_list=loader.train
    data_size = len(id_list)
    indices = np.arange(0,data_size)
    np.random.shuffle(indices)
    train_size = int(train_factor*data_size)
    val_size = int((1-train_factor)*data_size)

    ti = indices[:train_size]
    vi = indices[train_size:train_size+val_size]

    ti = ti.astype(int).tolist()
    vi = vi.astype(int).tolist()

    id_list = np.array(id_list)
    box2d_list = np.array(box2d_list)
    box3d_list = np.array(box3d_list)
    input_list = np.array(input_list)
    label_list = np.array(label_list)
    type_list = np.array(type_list)
    heading_list = np.array(heading_list)
    size_list = np.array(size_list)
    frustum_angle_list = np.array(frustum_angle_list)


    save_zipped_pickle([id_list[ti].tolist(),box2d_list[ti].tolist(),box3d_list[ti].tolist(),input_list[ti].tolist(),label_list[ti].tolist(),type_list[ti].tolist(),heading_list[ti].tolist(),size_list[ti].tolist(),frustum_angle_list[ti].tolist()],'sunrgbd_retina_train.pickle')
    save_zipped_pickle([id_list[vi].tolist(),box2d_list[vi].tolist(),box3d_list[vi].tolist(),input_list[vi].tolist(),label_list[vi].tolist(),type_list[vi].tolist(),heading_list[vi].tolist(),size_list[vi].tolist(),frustum_angle_list[vi].tolist()],'sunrgbd_retina_val.pickle')

split_dataset()


