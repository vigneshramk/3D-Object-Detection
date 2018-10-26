import torch.nn as nn
import torch
import time


class Model(nn.Module):
    def __init__(num_channels, activation=nn.ReLU, bn_decay=0):
        super().__init__()
        
        self.cnn1 = nn.Sequential(
                                nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(num_channels[1], momentum=1-bn_decay), #No momentum in original code. eps was None
                                self.Activation()
                                )
        self.cnn2 = nn.Sequential(
                                nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(num_channels[2], momentum=1-bn_decay), #No momentum in original code. eps was None
                                self.Activation()
                                )
        self.cnn3 = nn.Sequential(
                                nn.Conv2d(num_channels[2], num_channels[3], kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True), #32 bit floats in original code. padding was VALID
                                nn.BatchNorm2d(num_channels[3], momentum=1-bn_decay), #No momentum in original code. eps was None
                                self.Activation()
                                )
        self.layers = [self.cnn1, self.cnn2, self.cnn3]
        self.initializeWeights()

    def forward(self, x):
    	x = self.cnn1(x)
    	x = self.cnn2(x)
    	x = self.cnn3(x)
    	x = torch.max(x, dim=2, keepdim=True)[0]
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


    def save(self, fname="pointNetSetAbstraction_{}.pth".format(time.time())):
        torch.save(self.state_dict(), fname)
        return fname

    def load(self, fname):
        self.load_state_dict(torch.load(fname))