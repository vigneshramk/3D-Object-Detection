"""
  Regression PointNet(T-Net) for correcting the origin of the
  masked coordinate frame.
"""

import torch
import torch.nn as nn
import globalVariables as glb

class TNet(nn.Module):
    def __init__(self, ch=3):
        super(TNet, self).__init__()
        c1 = nn.Conv2d(ch, 128, (1, 1), stride=1, padding=0)
        c2 = nn.Conv2d(128, 128, (1, 1), stride=1, padding=0)
        c3 = nn.Conv2d(128, 256, (1, 1), stride=1, padding=0)
        rl = nn.ReLU(inplace=True)
        #mp = nn.MaxPool2d((ch, 1))

        bc1 = nn.BatchNorm2d(128)
        bc2 = nn.BatchNorm2d(128)
        bc3 = nn.BatchNorm2d(256)

        l1 = nn.Linear(266, 256)                 # TODO: Add Input dimension
        l2 = nn.Linear(256, 128)
        l3 = nn.Linear(128, 3)

        bl1 = nn.BatchNorm1d(256)
        bl2 = nn.BatchNorm1d(128)

        self.conv = nn.Sequential(c1, bc1, rl, c2, bc2, rl,
                                  c3, bc3, rl)

        self.fc = nn.Sequential(l1, bl1, rl, l2, bl2, rl, l3)
        self.initialization()

    def initialization(self):
        """ Xavier Initialization is suggested -- xavier uniform or normal? """
        for layer in self.conv:
            if not isinstance(layer, nn.Conv2d):
                continue
            try:
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0.0)
            except:
                pass

        for layer in self.conv:
            if not isinstance(layer, nn.Linear):
                continue
            try:
                nn.init.xavier_uniform_(layer.weight.data)
                nn.init.constant_(layer.bias.data, 0.0)
            except:
                pass


    def forward(self, point_cloud, one_hot_vec, logits):
        num_point = point_cloud.size(1)
        mask = logits[0:, 0:, 0:1] < logits[0:, 0:, 1:2] # BxNx1
        mask_count = torch.sum(mask, dim=1, keepdim=True).repeat(1, 1, 3) # Bx1x3
        point_cloud_xyz = point_cloud[0:, 0:, 0:3] # BxNx3

        mask_xyz_mean = torch.sum(mask.float().repeat(1, 1, 3)*point_cloud_xyz, dim=1, keepdim=True) # Bx1X3
        mask_xyz_mean = mask_xyz_mean/torch.max(mask_count, 1)[0].float().unsqueeze(1) # Bx1x3
        point_cloud_xyz_stage1 = point_cloud_xyz - mask_xyz_mean.repeat(1, num_point, 1)

        # ---- Regress 1st stage center ----
        x = point_cloud_xyz_stage1.permute(0, 2, 1)
        x = x.unsqueeze(3)

        # TODO: Input dimension for TNet is required
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, (num_point, 1))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, one_hot_vec), dim=1)
        stage1_center = self.fc(x)
        stage1_center = stage1_center + torch.squeeze(mask_xyz_mean, 1)

        return point_cloud_xyz, stage1_center, mask


