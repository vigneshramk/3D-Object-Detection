import pickle
import gzip

# Make a folder data inside the 3D-Object-Detection folder and put the pickle file inside it
SUN_PATH = './data'
SUN_FILE = 'sunrgbd_train_preprocessed.pickle'

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

def load_data(SUN_PATH, SUN_FILE):
    filename = SUN_PATH + '/' + SUN_FILE
    _,_, frustum_list, type_list, _,_=load_zipped_pickle(filename)
    classes = [type2class[l] for l in type_list]
    return frustum_list,classes

# # Testing the loader
# loader = SUNRGBD()
# frustum, classes = loader.train
# print(frustum[0].shape,classes[0])