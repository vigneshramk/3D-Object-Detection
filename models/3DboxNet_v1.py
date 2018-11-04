import torch.nn as nn
import torch
import time
import numpy as np
import globalVariables as glb

NUM_HEADING_BIN = glb.NUM_HEADING_BIN
NUM_SIZE_CLUSTER= glb.NUM_SIZE_CLUSTER

class Model(nn.Module):
    def __init__(num_in_channels=1, num_input_to_fc=(512+10), activation=nn.ReLU, bn_decay=0):
        super().__init__()
        self.Activation = activation
        self.cnn1 = nn.Sequential(
                                nn.Conv2d(num_in_channels, 128, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(128, momentum=1-bn_decay), #No momentum in original code. eps was None
                                self.Activation()
                                )
        self.cnn2 = nn.Sequential(
                                nn.Conv2d(128, 128, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(128, momentum=1-bn_decay), #No momentum in original code.
                                self.Activation()
                                )
        self.cnn3 = nn.Sequential(
                                nn.Conv2d(128, 256, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(256, momentum=1-bn_decay), #No momentum in original code.
                                self.Activation()
                                )
        self.cnn4 = nn.Sequential(
                                nn.Conv2d(256, 512, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(512, momentum=1-bn_decay), #No momentum in original code.
                                self.Activation()
                                )
        self.fc1  = nn.Sequential(
                                nn.Linear(num_input_to_fc, 512, bias=True),
                                nn.BatchNorm1d(512, momentum=1-bn_decay),
                                self.Activation()
                                )
        self.fc2  = nn.Sequential(
                                nn.Linear(512, 256, bias=True),
                                nn.BatchNorm1d(256, momentum=1-bn_decay),
                                self.Activation()
                                )
    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        self.fc3  = nn.Linear(256, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, bias=True)
        self.layers = [self.cnn1, self.cnn2, self.cnn3, self.cnn4, self.fc1, self.fc2, self.fc3]
        self.initializeWeights()
        self.end_points = {}
        self.mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))                                


    def forward(self, object_point_cloud, mask, stage1_center, one_hot_vec):
        self.end_points['stage1_center'] = stage1_center
        point_cloud_xyz_submean = object_point_cloud - stage1_center.unsqueeze(1)
        x = point_cloud_xyz_submean.unsqueeze(2)
        x = x.view(x.shape[0],x.shape[1],1,x.shape[2])
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        mask_expand = torch.repeat(mask.unsqueeze(-1), [1,1,1,512])
        masked_x = x*mask_expand
        x = nn.functional.max_pool2d(masked_x, kernel_size=(object_point_cloud.size(1),1), stride=(2,2), padding=0)
        x = torch.squeeze(x, dim=[1,2])
        x = torch.cat([x,one_hot_vec], 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        center = x[:,0:3] + stage1_center
        self.end_points['center']=center
        heading_scores = x[:,3:NUM_HEADING_BIN]
        heading_residuals_normalized = x[:3+NUM_HEADING_BIN,NUM_HEADING_BIN]
        self.end_points['heading_scores'] = heading_scores # BxNUM_HEADING_BIN
        self.end_points['heading_residuals_normalized'] = heading_residuals_normalized # BxNUM_HEADING_BIN (should be -1 to 1)
        self.end_points['heading_residuals'] = heading_residuals_normalized * (np.pi/NUM_HEADING_BIN) # BxNUM_HEADING_BIN
    
        size_scores = x[:,3+NUM_HEADING_BIN*2:NUM_SIZE_CLUSTER] # BxNUM_SIZE_CLUSTER
        size_residuals_normalized = x[:,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER:NUM_SIZE_CLUSTER*3]
        size_residuals_normalized = size_residuals_normalized.view([batch_size, NUM_SIZE_CLUSTER, 3]) # BxNUM_SIZE_CLUSTERx3
        end_points['size_scores'] = size_scores
        end_points['size_residuals_normalized'] = size_residuals_normalized
        end_points['size_residuals'] = size_residuals_normalized * self.mean_size_arr.unsqueeze(0)
        return end_points


    def initializeWeights(self, function=nn.init.xavier_normal_):
        for layer in self.layers:
            try:
                function(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0)
            except:
                try:
                    for l in layer:
                        try:
                            function(l.weight.data)
                            nn.init.constant_(l.bias.data, 0)
                        except:
                            pass
                except:
                    pass


    def save(self, fname="3DboxNet_v1_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))