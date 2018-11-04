import numpy as np
import Instance_3D_seg_v1
import torch
import pickle
import gzip

with gzip.open('..\\sunrgbd_train_preprocessed.p', 'rb') as f:
    id_list, box2d_list, input_list, type_list, frustum_angle_list, prob_list = pickle.load(f)

print("Loaded data!")
# print(type(type_list[0]))
# print(type(input_list))

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

num_inputs = len(type_list)
# print(num_inputs)

types_one_hot = np.zeros((num_inputs, 10))
for i in range(num_inputs):
	class_idx = type2class[type_list[i]]
	types_one_hot[i][class_idx] = 1

input_tensors = torch.FloatTensor(np.array(input_list[:16]))
types_one_hot = torch.ShortTensor(types_one_hot[:16])

input_tensors[:,:,3] = input_tensors[:,:,3]*0.299 + input_tensors[:,:,4]*0.587 + input_tensors[:,:,4]*0.114
input_tensors = input_tensors[:,:,:4]
print("Created tensors")

net = Instance_3D_seg_v1.InstanceSegNet(num_classes=10)
out = net(input_tensors, types_one_hot, batch_size=16)
print(out.shape)

# print(type(id_list[0]))
# print(type(box2d_list[0]))
# print(type(frustum_angle_list[0]))
# print(type(prob_list[0]))