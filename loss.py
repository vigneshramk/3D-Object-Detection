"""
  Corner Loss for Joint Optimization for Box Parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn

import numpy as np

def make_onehot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : shape(B, )
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable B x C.  One-hot encoded.
    '''
    one_hot = torch.FloatTensor(len(labels), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target

class CornerLoss(nn.module):
  def __init__(self):
    super(CornerLoss, self).__init__()
    self.num_heading_bin = 12
    self.num_size_cluster = 8

  def forward(self, mask_label, center_label, heading_class_label,
              heading_residual_label, size_class_label, size_residual_label,
              end_points, corner_loss_weight=10.0, box_loss_weight=1.0):

    # 3D Mask Loss
    mask_loss = fn.cross_entropy(end_points['mask_logits'], mask_label)

    # Huber Loss equivalent is smooth_L1_loss
    center_loss = fn.smooth_l1_loss(end_points['center'], center_label)

    stage1_center_loss = fn.smooth_l1_loss(end_points['stage1_center'], center_label)

    heading_class_loss = fn.cross_entropy(end_points['heading_scores'], heading_class_label)

    hcls_onehot = make_onehot(heading_class_label, self.num_heading_bin)
    heading_residual_normalized_label = heading_residual_label / (np.pi/self.num_heading_bin)
    heading_loss_input = torch.sum(end_points['heading_residuals_normalized']*hcls_onehot, dim=1)
    heading_residual_normalized_loss = fn.smooth_l1_loss(heading_loss_input, heading_residual_normalized_label)

    size_class_loss = fn.cross_entropy(end_points['size_scores'], size_class_label)

    scls_one_hot = make_onehot(size_class_label, self.num_size_cluster)
    scls_one_hot_tiled = torch.unsqueeze(scls_one_hot, 2).repeat(1, 1, 3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*scls_one_hot_tiled, dim=1)


    mean_size_arr_expand = torch.unsqueeze(torch.from_numpy(g_mean_size_arr), 0)
    mean_size_label = torch.sum(scls_onehot_tiled * mean_size_arr_expand, dim=1)
    size_residual_label_normalized = size_residual_label / mean_size_label
    size_residual_normalized_loss = fn.smooth_l1_loss(size_residual_label_normalized, predicted_size_residual_normalized)

    corners_3d = self.get_box3d_corners(end_points['center'], end_points['heading_residuals'],
                                        end_points['size_residuals'])
    gt_mask = torch.unsqueeze(hcls_onehot, 2).repeat(1, 1, self.num_size_cluster)*
              torch.unsqueeze(hcls_onehot, 1).repeat(1, self.num_heading_bin, 1)
    coreners_3d_pred = torch_sum(torch.unsqueeze(torch.unsqueeze(gt_mask, -1), -1).float() * corners_3d, dim=(1, 2))

    heading_bin_centers = torch.from_numpy(np.arange(0, 2*np.pi, 2*np.pi/self.num_heading_bin)).float()
    heading_label = torch.unsqueeze(heading_residual_label, 1) + torch.unsqueeze(heading_bin_centers, 0)
    heading_label = torch.sum(hcls_onehot.float() * heading_label, dim=1)

    mean_size = torch.unsqueeze(torch.from_numpy(g_mean_size_arr).float(), 0)
    size_label = mean_sizes + torch.unsqueeze(size_residual_label, 1)
    size_label = torch.sum(torch.unsqueeze(scls_onehot.float(), -1)*size_label, dim=1)
    corners_3d_gt = self.get_box3d_corners_helper(center_label, heading_label, size_label)
    corners_3d_gt_flip = self.get_box3d_corners_helper(center_label, heading_label+np.pi, size_label)

    center_loss = torch.min(fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt), fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt_flip))

    # Weighted sum of all losses
    loss = mask_loss + box_loss_weight * (center_loss + heading_class_loss + size_class_loss + \
           heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + \
           stage1_center_loss + corner_loss_weight*corners_loss)

    return loss

    def get_box3d_corners(self, center, heading_residuals, size_residuals):
      """ TF layer.
      Inputs:
          center: (B,3)
          heading_residuals: (B,NH)
          size_residuals: (B,NS,3)
      Outputs:
          box3d_corners: (B,NH,NS,8,3) tensor
      """
      batch_size = center.shape[0]
      heading_bin_centers = torch.from_numpy(np.arange(0, 2*np.pi, 2*np.pi/self.num_heading_bin)).float() # (NH,)
      headings = heading_residuals + torch.unsqueeze(heading_bin_centers, 0) # (B,NH)

      mean_sizes = torch.unsqueeze(torch.from_numpy(g_mean_size_arr).float(), 0) + size_residuals # (B,NS,1)
      sizes = mean_sizes + size_residuals # (B,NS,3)
      size = torch.unsqueeze(sizes, 1).repeat(1, self.num_heading_bin, 1, 1)
      headings = torch.unsqueeze(headings, -1).repeat(1, 1, self.num_size_cluster)
      centers = torch.unsqueeze(torch.unsqueeze(center, 1), 1).repeat(1, self.num_heading_bin, self.num_size_cluster, 1)

      N = batch_size * self.num_heading_bin * self.num_size_cluster
      corners_3d = self.get_box3d_corners_helper(centers.view(N, 3), headings.view(N), sizes.view(N, 3))

      return corners_3d.view(batch_size, self.num_heading_bin, self.num_size_cluster, 8, 3)


    def get_box3d_corners_helper(centers, headings, sizes):
      """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
      #print '-----', centers
      N = centers.shape[0]
      l = sizes[:, 1]
      w = sizes[:, 1]
      h = sizes[:, 2]
      x_corners = torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], axis=1) # (N,8)
      y_corners = torch.cat([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], axis=1) # (N,8)
      z_corners = torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], axis=1) # (N,8)
      corners = torch.cat([torch.unsqueeze(x_corners, 1), torch.unsqueeze(y_corners, 1), torch.unsqueeze(z_corners, 1)], axis=1) # (N,3,8)
      c = torch.cos(headings)
      s = torch.sin(headings)
      ones = torch.ones([N]).float()
      zeros = torch.zeros([N]).float()
      row1 = torch.cat([c, zeros, s], dim=1) # (N,3)
      row2 = torch.cat([zeros, ones, zeros], dim=1)
      row3 = torch.cat([-s, zeros, c], dim=1)
      R = torch.cat([torch.unsqueeze(row1, 1), torch.unsqueeze(row2, 1), torch.unsqueeze(row3, 1)], dim=1) # (N,3,3)
      corners_3d = torch.matmul(R, corners) # (N,3,8)
      corners_3d += torch.unsqueeze(centers, 2).repeat(1, 1, 8) # (N,3,8)
      corners_3d = corners_3d.permute(0, 2, 1) # (N,8,3)
      return corners_3d


