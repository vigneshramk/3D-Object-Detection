import pickle
import gzip
import numpy as np

# Make a folder data inside the 3D-Object-Detection folder and put the pickle file inside it
SUN_PATH = './data'
SUN_FILE = 'sunrgbd_3d.pickle'

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SUNRGBD:

    def __init__(self):
        self.dev_set = None
        self.train_set = None
        self.test_set = None
  
    @property
    def dev(self):
        pass

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_data(SUN_PATH, SUN_FILE)
        return self.train_set
  
    @property
    def test(self):
        pass

def get_frustum(input):

    result = []
    for i in range(len(input)):
        data = input[i]
        xyz = data[:,:3]
        r,g,b = data[:,3],data[:,4],data[:,5]
        intensity = 0.299*r + 0.587*g + 0.114*b
        intensity = np.expand_dims(intensity,axis=1)
        curr = np.hstack((xyz,intensity))
        result.append(curr)

    return result

def load_data(SUN_PATH, SUN_FILE):
    filename = SUN_PATH + '/' + SUN_FILE
    id_list,box2d_list,box3d_list,input_list,label_list,type_list,heading_list,box3d_size_list,frustum_angle_list=load_zipped_pickle(filename)

    class_list = [type2class[l] for l in type_list]
    frustum_list = get_frustum(input_list)

    return id_list,box2d_list,box3d_list,frustum_list,label_list,type_list,heading_list,box3d_size_list,frustum_angle_list
    # return frustum_list,class_list,label_list,box3d_list

# Testing the loader
def test_loader():
    loader = SUNRGBD()
    frustum,class_label,seg_mask,box3d = loader.train
    for i in range(len(frustum)):
        print(frustum[i].shape,class_label[i],seg_mask[i].shape,box3d[i].shape)

# Call this to test
# test_loader()