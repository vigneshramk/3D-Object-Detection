from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import sys
import torch
from loader import SUNRGBD

class SUN_TrainDataSet(Dataset):

    def __init__(self):
        loader = SUNRGBD()
        self.frustum_list,self.class_list,self.label_list,self.box3d_list = loader.train

    def __len__(self):
        return len(self.frustum_list)

    def __getitem__(self,index):
        
        frustum,class_label,seg_mask,box3d = self.frustum_list[index], self.class_list[index],self.label_list[index],self.box3d_list[index]
        frustum = torch.FloatTensor(frustum)
        seg_mask,box3d = torch.LongTensor(seg_mask),torch.LongTensor(box3d)
        return (frustum,class_label,seg_mask,box3d)

def convert_batch(batch):

	batch_size = len(batch)

	max_points = float('-inf')
	for i in range(batch_size):
		max_points = max(max_points,batch[i][0].size(0))

	frustum_batch = torch.zeros(batch_size,max_points,4)
	class_batch = []
	seg_batch = torch.zeros(batch_size,max_points)
	box3d_batch = torch.zeros(batch_size,8,3)

	for x in range(batch_size):

		frustum,class_label,seg_mask,box3d = batch[x]
		N_curr = frustum.shape[0]
		frustum_batch[x][:N_curr][:] = frustum
		seg_batch[x][:N_curr] = seg_mask
		
		class_batch.append(class_label)
		box3d_batch[x][:][:] = box3d

	class_batch = torch.IntTensor(class_batch)

	return frustum_batch,class_batch,seg_batch,box3d_batch
		
class SUN_TrainLoader(DataLoader):

    def __init__(self,*args,**kwargs):
        super(SUN_TrainLoader, self).__init__(*args, **kwargs)
        self.collate_fn = convert_batch


def test_dataloader():
	train_dataset = SUN_TrainDataSet()
	train_loader = SUN_TrainLoader(train_dataset, batch_size=5, shuffle=True,num_workers=1, pin_memory=False)

	for data in train_loader:

		frustum_batch,class_batch,seg_batch,box3d_batch = data

		print(frustum_batch.shape,class_batch.shape,seg_batch.shape,box3d_batch.shape)

# Run this to test
# test_dataloader()

