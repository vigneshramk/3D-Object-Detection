import numpy as np
import Instance_3D_seg_v1
import TNet
import torch
import pickle
import gzip
import ThreeDboxNet_v1 as boxNet
import loss

# with gzip.open('..\\sunrgbd_train_preprocessed.p', 'rb') as f:
#     id_list, box2d_list, input_list, type_list, frustum_angle_list, prob_list = pickle.load(f)

# print("Loaded data!")
# # print(type(type_list[0]), type(input_list), type(id_list[0]))
# # print(type(box2d_list[0]), type(frustum_angle_list[0]), type(prob_list[0]))

# # Dictionary for converting string labels into one hot vectors
# type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

# num_inputs = len(type_list)    # The number of data points
# # print(num_inputs)

# types_one_hot = np.zeros((num_inputs, 10))
# for i in range(num_inputs):
#   class_idx = type2class[type_list[i]]
#   types_one_hot[i][class_idx] = 1

# # mini_input = np.zeros((16,2048,6))
# # for i in range(16):
# # 	mini_input[i,:,:] = input_list[i + 16]
# # print(mini_input.shape)
# # np.save('mini_input.npy', mini_input)

# # mini_labels = types_one_hot[16:32]
# # np.save('..\\mini_labels.npy', mini_labels)

mini_input = np.load('../mini_input.npy')
mini_labels = np.load('../mini_labels.npy')

mini_input[:,:,3] = mini_input[:,:,3]*0.299 + mini_input[:,:,4]*0.587 + mini_input[:,:,4]*0.114
mini_input = mini_input[:,:,:4]
input_tensors = torch.FloatTensor(mini_input)
types_one_hot = torch.FloatTensor(mini_labels)
print("Created tensors")

# print(input_tensors.size())
# print(types_one_hot.size())

net = Instance_3D_seg_v1.InstanceSegNet(num_classes=10)
out = net(input_tensors, types_one_hot)

tnet = TNet.TNet(3)
pc_xyz, stage1_center, mask = tnet(input_tensors, types_one_hot, out)
box = boxNet.Model(num_in_channels=3, num_input_to_fc=(512+10))
end_points = box.forward(pc_xyz, mask, stage1_center, types_one_hot)

corner_loss = loss.CornerLoss_sunrgbd()
mask_label = torch.zeros((16, 2048)).int()
center_label = torch.zeros((16, 3))
heading_class_label = torch.zeros((16)).int()
heading_residual_label = torch.zeros((16))
size_class_label = torch.zeros((16))
size_residual_label = torch.zeros((16, 3))
loss_vall = corner_loss(out, mask_label, center_label, heading_class_label,
                        heading_residual_label, size_class_label, size_residual_label, end_points)
