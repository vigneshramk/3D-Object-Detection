import torch.nn as nn
import torch
import time
import Instance_3D_seg_v1 as ThreeDseg
import ThreeDboxNet_v1 as boxNet
from TNet import TNet

class Model(nn.Module):
    def __init__():
        super().__init__()
        self.ThreeDseg = ThreeDseg.InstanceSegNet()
        self.TNet = TNet()
        self.boxNet = boxNet.Model(num_in_channels=3, num_input_to_fc=(512+10), activation=nn.ReLU, bn_decay=0)


    def forward(self, point_cloud, one_hot_vec):
        logits = self.ThreeDseg.forward(point_cloud,one_hot_vec)
        point_cloud_xyz, stage1_center, mask = self.TNet.forward(point_cloud,one_hot_vec,logits)
        end_points = self.boxNet.forward(point_cloud_xyz, mask, stage1_center, one_hot_vec)
        return logits, end_points


    def save(self, fname="Mother_v1_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))