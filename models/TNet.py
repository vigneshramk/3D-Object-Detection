"""
  Regression PointNet(T-Net) for correcting the origin of the
  masked coordinate frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

class TNet(nn.Module):
  def __init__(self, ch):
    super(TNet, self).__init__()
    c1 = nn.Conv2d(ch, 128, (1, 1), stride=1, padding=0)
    c2 = nn.Conv2d(128, 128, (1, 1), stride=1, padding=0)
    c3 = nn.Conv2d(128, 256, (1, 1), stride=1, padding=0)
    rl = nn.ReLU(inplace=True)
    mp = nn.MaxPool2d((ch, 1))

    bc1 = nn.BatchNorm2d(128)
    bc2 = nn.BatchNorm2d(128)
    bc3 = nn.BatchNorm2d(256)

    l1 = nn.Linear(, 256)
    l2 = nn.Linear(256, 128)
    l3 = nn.Linear(128, 3)

    bl1 = nn.BatchNorm1d(256)
    bl2 = nn.BatchNorm1d(128)

    self.conv = nn.sequential(c1, bc1, rl, c2, bc2, rl,
                              c3, bc3, rl, mp)

    self.fc = nn.sequential(l1, bl1, rl, l2, bl2, rl, l3)
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


  def forward(self, x, ob_type=None):
    x = self.conv(x)
    x = x.view(x.size(0), -1)
    if ob_type is not None:
      x = torch.cat((x, ob_type), dim=1)
    predicted_center = self.fc(x)

    return predicted_center


