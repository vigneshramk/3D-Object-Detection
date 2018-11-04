'''
Frustum PointNets: 3D Instance Segmentation PointNet (v1) Model
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import globalVariables as glb

class InstanceSegNet(nn.Module):
  '''
  3D instance segmentation PointNet v1 network.
  Input:
    num_classes: Indicates length of a instance of one-hot vector (Scalar)
    use_xavier: Flag for indicating whether to use Xavier or Kaiming initialization for model weights (Boolean scalar)
    bn_decay: Indicates proportion of population running mean stats kept in batchnorm layers (Float scalar)

    point_cloud: Tensor of shape (Batch_size,4,Num_points)
      Frustum point clouds with XYZ and intensity in point channels
      XYZs are in frustum coordinate
    one_hot_vec: Tensor of shape (Batch_size,3)
      Vectors indicating predicted object type
    batch_size: Batch size for data
    is_training: Flag for indicating whether model is being used for training (Boolean scalar)

  Output:
    logits: Tensor of shape (Batch_size,2,Num_points), scores for background/clutter and object
  '''
  def __init__(self, num_classes = 10, use_xavier = True, bn_decay = None):
    super(InstanceSegNet, self).__init__()
    bn_momentum = (1 - bn_decay) if bn_decay is not None else 0.1

    self.conv1 = nn.Conv2d(4, 64, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn1 = nn.BatchNorm2d(64, momentum = bn_momentum)

    self.conv2 = nn.Conv2d(64, 64, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn2 = nn.BatchNorm2d(64, momentum = bn_momentum)

    self.conv3 = nn.Conv2d(64, 64, [1,1], stride=(1, 1), padding=(0,0))
    self.bn3 = nn.BatchNorm2d(64, momentum = bn_momentum)

    self.conv4 = nn.Conv2d(64, 128, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn4 = nn.BatchNorm2d(128, momentum = bn_momentum)

    self.conv5 = nn.Conv2d(128, 1024, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn5 = nn.BatchNorm2d(1024, momentum = bn_momentum)

    self.conv6 = nn.Conv2d((1088 + num_classes), 512, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn6 = nn.BatchNorm2d(512, momentum = bn_momentum)

    self.conv7 = nn.Conv2d(512, 256, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn7 = nn.BatchNorm2d(256, momentum = bn_momentum)

    self.conv8 = nn.Conv2d(256, 128, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn8 = nn.BatchNorm2d(128, momentum = bn_momentum)

    self.conv9 = nn.Conv2d(128, 128, (1, 1), stride=(1, 1), padding=(0,0))
    self.bn9 = nn.BatchNorm2d(128, momentum = bn_momentum)
    self.dp1 = nn.Dropout2d(p=0.5)

    self.conv10 = nn.Conv2d(128, 2, (1, 1), stride=(1, 1), padding=(0,0))

    # Handling weight & bias intialization explicitly
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if use_xavier:
          nn.init.xavier_uniform_(m.weight)
        else:
          nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)

      ### CHECK WITH TENSORFLOW INITIALIZATION
      # elif isinstance(m, nn.BatchNorm2d):
        # nn.init.constant_(m.weight, 1)
        # nn.init.constant_(m.bias, 0)

  def forward(self, point_cloud, one_hot_vec):
    batch_size = point_cloud.size()[0]
    point_cloud = point_cloud.permute(0, 2, 1) # 3D Tensor: B x N x C -> B x C x N
    num_points = point_cloud.size()[2]

    x = torch.unsqueeze(point_cloud, 3)        # 4D Tensor: B x C x N x 1

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    point_feat = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(point_feat)))
    x = F.relu(self.bn5(self.conv5(x)))

    global_feat = torch.unsqueeze(torch.mean(x, dim=2), 2)              # Output Tensor size: B x 64 x 1 x 1
    # global_feat = F.max_pool2d(x, [num_points,1], stride = [2,2], padding=0)    # Output Tensor size: B x 64 x 1 x 1
    one_hot_vec = torch.unsqueeze(torch.unsqueeze(one_hot_vec, 2), 2)           # 4D Tensor: B x K x 1 x 1
    global_feat = torch.cat([global_feat, one_hot_vec], dim=1)          # Concatenated Tensor size: B x (64+K) x 1 x 1
    global_feat = global_feat.repeat(1,1,num_points,1)                  # Resulting Tensor size: B x (64+K) x N x 1
    concat_feat = torch.cat([point_feat, global_feat], dim=1)    # Concatenated Tensor size: B x (1024+(64+K)) x N x 1

    x = F.relu(self.bn6(self.conv6(concat_feat)))
    x = F.relu(self.bn7(self.conv7(x)))
    x = F.relu(self.bn8(self.conv8(x)))
    x = F.relu(self.bn9(self.conv9(x)))
    x = self.dp1(x)

    logits = self.conv10(x)
    logits = torch.squeeze(logits, 3)        # Final output Tensor size: B x 2 x N

    return logits.permute(0, 2, 1)
