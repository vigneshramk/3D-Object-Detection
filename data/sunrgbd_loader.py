from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import torch
from data.loader import SUNRGBD
import data.data_utils as data_utils

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

class SUN_TrainDataSet(Dataset):
    def __init__(self, npoints, random_flip=False, random_shift=False, rotate_to_center=False, overwritten_data_path=None, from_rgb_detection=False, one_hot=True):
        
        loader = SUNRGBD()
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        # if overwritten_data_path is None:
        #     overwritten_data_path = os.path.join(BASE_DIR, '%s_1002.zip.pickle'%(split))

        self.from_rgb_detection = from_rgb_detection
        
        self.id_list,self.box2d_list,self.box3d_list,self.input_list,self.label_list,self.type_list,self.heading_list,self.size_list,self.frustum_angle_list=loader.train

    def __len__(self):
            return len(self.input_list)

    def __getitem__(self, index):
        # ------------------------------ INPUTS ----------------------------
        rot_angle = self.get_center_view_rot_angle(index)

        # compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert(cls_type in ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'])
            # one_hot_vec = np.zeros((NUM_CLASS))
            class_label = type2onehotclass[cls_type]

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        
        # Resample - Should we do this?
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        if self.from_rgb_detection:
            if self.one_hot:
                return point_set, rot_angle, self.prob_list[index], one_hot_vec
            else:
                return point_set, rot_angle, self.prob_list[index]
        
        # Return image id
        image_id = self.id_list[index]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index] 
        seg = seg[choice]

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
        else:
            box3d_center = self.get_box3d_center(index)

        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = data_utils.size2class(self.size_list[index], self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            if np.random.random()>0.5:
                point_set[:,0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
                # NOTE: rot_angle won't be correct if we have random_flip...
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0]**2+box3d_center[1]**2))
            shift = np.clip(np.random.randn()*dist*0.05, dist*0.8, dist*1.2)
            point_set[:,2] += shift
            box3d_center[2] += shift
            height_shift = np.random.random()*0.4-0.2 # randomly shift +-0.2 meters
            point_set[:,1] += height_shift
            box3d_center[1] += height_shift

        angle_class, angle_residual = data_utils.angle2class(heading_angle, NUM_HEADING_BIN)

        point_set = torch.FloatTensor(point_set)
        seg = torch.FloatTensor(seg)
        box3d_center = torch.FloatTensor(box3d_center)
        size_residual = torch.FloatTensor(size_residual)

        if self.one_hot:
            return image_id, point_set, seg, box3d_center, angle_class, angle_residual, size_class, size_residual, rot_angle, class_label
        else:
            return image_id, point_set, seg, box3d_center, angle_class, angle_residual, size_class, size_residual, rot_angle

    def get_center_view_rot_angle(self, index):
        return np.pi/2.0 + self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        box3d_center = (self.box3d_list[index][0,:] + self.box3d_list[index][6,:])/2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        box3d_center = (self.box3d_list[index][0,:] + self.box3d_list[index][6,:])/2.0
        return data_utils.rotate_pc_along_y(np.expand_dims(box3d_center,0), self.get_center_view_rot_angle(index)).squeeze()
        
    def get_center_view_box3d(self, index):
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return data_utils.rotate_pc_along_y(box3d_center_view, self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Input ps is NxC points with first 3 channels as XYZ
            z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return data_utils.rotate_pc_along_y(point_set, self.get_center_view_rot_angle(index))



def convert_batch(batch):

    batch_size = len(batch)

    max_points = float('-inf')
    for i in range(batch_size):
        max_points = max(max_points,batch[i][0].size(0))

    frustum_batch = torch.zeros(batch_size,max_points,4)
    class_batch = []
    seg_batch = torch.zeros(batch_size,max_points)

    labels_dict = {}
    

    image_id_batch = []
    box3d_center_batch = []
    angle_class_batch = []
    angle_residual_batch = []
    size_class_batch = []
    size_residual_batch = []
    rot_angle_batch = []

    for x in range(batch_size):

        image_id, point_set, seg, box3d_center, angle_class, angle_residual, size_class, size_residual, rot_angle, class_label = batch[x]

        image_id_batch.append(image_id)

        N_curr = point_set.shape[0]
        frustum_batch[x][:N_curr][:] = point_set
        seg_batch[x][:N_curr] = seg
        class_batch.append(class_label)
        
        box3d_center_batch.append(box3d_center)
        
        angle_class_batch.append(angle_class)
        angle_residual_batch.append(angle_residual)
        
        size_class_batch.append(size_class)
        size_residual_batch.append(size_residual)

        # rot_angle_batch.append(rot_angle)

    image_id_batch = torch.IntTensor(image_id_batch)

    box3d_center_batch = torch.stack(box3d_center_batch)

    angle_class_batch = torch.IntTensor(angle_class_batch)
    angle_residual_batch = torch.FloatTensor(angle_residual_batch)
    size_class_batch = torch.IntTensor(size_class_batch)
    size_residual_batch = torch.stack(size_residual_batch)
    # rot_angle_batch = torch.stack(rot_angle_batch)

    class_batch = torch.IntTensor(class_batch)

    labels_dict['mask_label'] = seg_batch
    labels_dict['center_label'] = box3d_center_batch
    labels_dict['heading_class_label'] =  angle_class_batch
    labels_dict['heading_residual_label'] = angle_residual_batch
    labels_dict['size_class_label'] = size_class_batch
    labels_dict['size_residual_label'] = size_residual_batch

    return image_id_batch,frustum_batch,class_batch,labels_dict
        
class SUN_TrainLoader(DataLoader):

    def __init__(self,*args,**kwargs):
        super(SUN_TrainLoader, self).__init__(*args, **kwargs)
        self.collate_fn = convert_batch


def test_dataloader():
    train_dataset = SUN_TrainDataSet(2048)
    train_loader = SUN_TrainLoader(train_dataset, batch_size=5, shuffle=True,num_workers=1, pin_memory=False)

    i = 0
    for data in train_loader:

        if i>0:
            break

        image_id_batch,frustum_batch,class_batch,labels_dict = data

        print(frustum_batch.shape,class_batch.shape)
        print(labels_dict.keys())
        i+=1

def dataloader():
    train_dataset = SUN_TrainDataSet(2048)
    train_loader = SUN_TrainLoader(train_dataset, batch_size=16, shuffle=True)
    return train_loader
# Run this to test
# test_dataloader()

