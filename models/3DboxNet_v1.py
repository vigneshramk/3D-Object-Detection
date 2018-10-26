import torch.nn as nn
import torch
import time

NUM_HEADING_BIN =
NUM_SIZE_CLUSTER=

class Model(nn.Module):
    def __init__(num_in_channels, num_pool, num_input_to_fc, activation=nn.ReLU, bn_decay=0):
        super().__init__()
        self.num_point=num_pool
        self.Activation = activation
        self.cnn1 = nn.Sequential(
                                nn.Conv2d(num_in_channels, 128, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(128, eps=bn_decay, momentum=0.1), #No momentum in original code. eps was None
                                self.Activation()
                                )
        self.cnn2 = nn.Sequential(
                                nn.Conv2d(128, 128, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(128, eps=bn_decay, momentum=0.1), #No momentum in original code.
                                self.Activation()
                                )
        self.cnn3 = nn.Sequential(
                                nn.Conv2d(128, 256, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(256, eps=bn_decay, momentum=0.1), #No momentum in original code.
                                self.Activation()
                                )
        self.cnn4 = nn.Sequential(
                                nn.Conv2d(256, 512, (1,1), (1,1), (0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(512, eps=bn_decay, momentum=0.1), #No momentum in original code.
                                self.Activation()
                                )
        self.pool = nn.MaxPool2d(kernel_size=(self.num_point,1), stride=(2,2), padding=0) #padding was VALID
        self.fc1  = nn.Sequential(
                                nn.Linear(num_input_to_fc, 512, bias=True),
                                nn.BatchNorm1d(512, eps=bn_decay, momentum=0.1),
                                self.Activation()
                                )
        self.fc2  = nn.Sequential(
                                nn.Linear(512, 256, bias=True),
                                nn.BatchNorm1d(256, eps=bn_decay, momentum=0.1),
                                self.Activation()
                                )
    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
        self.fc3  = nn.Linear(256, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, bias=True)
        self.layers = [self.cnn1, self.cnn2, self.cnn3, self.cnn4, self.fc1, self.fc2, self.fc3]
        self.initializeWeights()
                                


    def forward(self, object_point_cloud, one_hot_vec):
        x = object_point_cloud.view(object_point_cloud.shape[0],object_point_cloud.shape[1],1,object_point_cloud.shape[2])
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=[1,2])
        x = torch.cat([x,one_hot_vec], 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

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