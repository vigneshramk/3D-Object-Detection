import torch.nn as nn
import torch
import time
import pointNetSetAbstraction 

class Model(nn.Module):
    def __init__(num_in_channels, num_input_to_fc, activation=nn.ReLU, bn_decay=0):
        super().__init__()
        
        self.set1 = pointNetSetAbstraction.Model(num_channels=[num_in_channels,64,64,128])
        self.set2 = pointNetSetAbstraction.Model(num_channels=[128,128,128,256])
        self.set3 = pointNetSetAbstraction.Model(num_channels=[256,256,256,512])
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
        self.layers = [self.set1, self.set2, self.set3, self.fc1, self.fc2, self.fc3]
        self.initializeWeights()

    def forward(self, object_point_cloud, one_hot_vec):
    	x = object_point_cloud
        x = self.set1(x)
    	x = self.set2(x)
    	x = self.set3(x)
    	x = x.view(x.size(1),-1) #batch_size X -1
    	x = torch.cat([x, one_hot_vec], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


    def initializeWeights(self, function=nn.init.xavier_normal_):
        for layer in self.layers:
            try:
                layer.initializeWeights(function)
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
            except:
                pass            

    def save(self, fname="3DboxNet_v2_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))