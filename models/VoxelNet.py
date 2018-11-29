import torch
import torch.nn as nn
import torch.nn.functional as F
import models.globalVariables as glb

class Conv2dV(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True, batch_norm=True):
        super(Conv2dV, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if activation:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self,x):
        x = self.conv(x)

        if self.bn is not None:
            x=self.bn(x)

        if self.act is not None:
            return self.act(x)

        return x

# conv3d + bn + relu
class Conv3dV(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3dV, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.act = nn.ReLU(inplace=True)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return self.act(x)

# Fully Connected Network
class FCN(nn.Module):

    def __init__(self,cin,cout):
        super(FCN, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        # KK is the stacked k across batch
        kk, t, _ = x.shape
        x = self.linear(x.view(kk*t,-1))
        x = self.act(self.bn(x))
        return x.view(kk, t, -1)

# Voxel Feature Encoding layer
class VFE(nn.Module):

    def __init__(self, cin, cout):
        super(VFE, self).__init__()
        assert cout % 2 == 0

        self.units = cout // 2
        self.fcn = FCN(cin, self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)

        #locally aggregated feature
        laf = torch.max(pwf, 1)[0].unsqueeze(1).repeat(1, 2048, 1)

        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=2)

        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units*2)
        pwcf = pwcf * mask.float()
        return pwcf

# Stacked Voxel Feature Encoding
class SVFE(nn.Module):

    def __init__(self):
        super(SVFE, self).__init__()
        self.vfe_1 = VFE(4,32)
        self.vfe_2 = VFE(32,128)
        self.fcn = FCN(128,128)

    def forward(self, x):
        mask = torch.ne(torch.max(x,2)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)

        return x.transpose(2, 1).unsqueeze(-1)

# Convolutional Middle Layer
class CML(nn.Module):
    def __init__(self):
        super(CML, self).__init__()
        self.conv3d_1 = Conv2dV(128, 64, 1, s=1, p=0)
        self.conv3d_2 = Conv2dV(64, 64, 1, s=1, p=0)
        self.conv3d_3 = Conv2dV(64, 64, 1, s=1, p=0)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.block_1 = [Conv2dV(64, 128, 1, 1, 0)]
        self.block_1 += [Conv2dV(128, 128, 1, 1, 0) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2dV(128, 128, 1, 1, 0)]
        self.block_2 += [Conv2dV(128, 128, 1, 1, 0) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2dV(128, 256, 1, 1, 0)]
        self.block_3 += [nn.Conv2d(256, 256, 1, 1, 0) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256))
        #self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 3, 1, 0), nn.BatchNorm2d(256))

        self.score_head = Conv2dV(256, 2, (1, 3), 1, 0, activation=False, batch_norm=False)

    def forward(self,x):
        x = self.block_1(x)
        x_skip_1 = x

        x = self.block_2(x)
        x_skip_2 = x

        x = self.block_3(x)

        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)

        x = torch.cat((x_0,x_1,x_2), dim=-1)

        return self.score_head(x)

class VoxelNet(nn.Module):
    def __init__(self):
        super(VoxelNet, self).__init__()
        self.svfe = SVFE()
        self.cml = CML()
        self.seg = SegNet()

    def forward(self, point_cloud, one_hot_vec):
        point_cloud = self.augument_point_cloud(point_cloud)

        # feature learning network
        vwfs = self.svfe(point_cloud)

        # convolutional middle network
        cml_out = self.cml(vwfs)

        #point_cloud = point_cloud.permute(0, 2, 1) # 3D Tensor: B x N x C -> B x C x N
        #point_cloud = torch.unsqueeze(point_cloud, 3)        # 4D Tensor: B x C x N x 1
        #logits = torch.squeeze(self.seg(point_cloud))
        logits = torch.squeeze(self.seg(cml_out))

        return logits.transpose(2, 1)

   def augument_point_cloud(self, point_cloud):
       mean = torch.mean(point_cloud[:, :, :-1], dim=1)
       mean = mean.unsqueeze(1).expand(-1, point_cloud.size(1), -1)
       return torch.cat((point_cloud, point_cloud[:, :, :-1] - mean), dim=2)
