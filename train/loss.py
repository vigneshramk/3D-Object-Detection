"""
    Corner Loss for Joint Optimization for Box Parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import models.globalVariables as glb
from scipy.spatial import ConvexHull

NUM_HEADING_BIN = glb.NUM_HEADING_BIN
NUM_SIZE_CLUSTER = glb.NUM_SIZE_CLUSTER
mean_size_arr = glb.mean_size_arr

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
    target = one_hot.cuda().scatter_(1, labels.unsqueeze(-1).long(), 1)

    return target

def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
     Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.shape[0]
    heading_bin_centers = torch.from_numpy(np.arange(0, 2*np.pi, 2*np.pi/NUM_HEADING_BIN)).float() # (NH,)
    headings = heading_residuals + torch.unsqueeze(heading_bin_centers, 0).cuda() # (B,NH)

    mean_sizes = torch.unsqueeze(mean_size_arr.float(), 0) + size_residuals # (B,NS,1)
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = torch.unsqueeze(sizes, 1).repeat(1, NUM_HEADING_BIN, 1, 1)
    headings = torch.unsqueeze(headings, -1).repeat(1, 1, NUM_SIZE_CLUSTER)
    centers = torch.unsqueeze(torch.unsqueeze(center, 1), 1).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)

    N = batch_size * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N, 3), headings.view(N), sizes.view(N, 3))

    return corners_3d.view(batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = centers.shape[0]
    l = sizes[:, 0:1]
    w = sizes[:, 1:2]
    h = sizes[:, 2:3]
    x_corners = torch.cat([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], dim=1) # (N,8)
    corners = torch.cat([torch.unsqueeze(x_corners, 1), torch.unsqueeze(y_corners, 1), torch.unsqueeze(z_corners, 1)], dim=1) # (N,3,8)
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = torch.ones([N]).float().cuda()
    zeros = torch.zeros([N]).float().cuda()
    row1 = torch.stack([c, zeros, s], dim=1) # (N,3)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-s, zeros, c], dim=1)
    R = torch.cat([torch.unsqueeze(row1, 1), torch.unsqueeze(row2, 1), torch.unsqueeze(row3, 1)], dim=1) # (N,3,3)
    corners_3d = torch.matmul(R, corners) # (N,3,8)
    corners_3d += torch.unsqueeze(centers, 2).repeat(1, 1, 8) # (N,3,8)
    corners_3d = corners_3d.permute(0, 2, 1) # (N,8,3)
    return corners_3d


class CornerLoss(nn.Module):
    def __init__(self):
        super(CornerLoss, self).__init__()

    def forward(self, mask_label, center_label, heading_class_label,
                heading_residual_label, size_class_label, size_residual_label,
                end_points, corner_loss_weight=10.0, box_loss_weight=1.0):

        # 3D Mask Loss
        mask_loss = fn.cross_entropy(end_points['mask_logits'], mask_label)

        # Huber Loss equivalent is smooth_L1_loss
        center_loss = fn.smooth_l1_loss(end_points['center'], center_label)

        stage1_center_loss = fn.smooth_l1_loss(end_points['stage1_center'], center_label)

        heading_class_loss = fn.cross_entropy(end_points['heading_scores'], heading_class_label)

        hcls_onehot = make_onehot(heading_class_label, NUM_HEADING_BIN)
        heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
        heading_loss_input = torch.sum(end_points['heading_residuals_normalized']*hcls_onehot, dim=1)
        heading_residual_normalized_loss = fn.smooth_l1_loss(heading_loss_input, heading_residual_normalized_label)

        size_class_loss = fn.cross_entropy(end_points['size_scores'], size_class_label)

        scls_one_hot = make_onehot(size_class_label, NUM_SIZE_CLUSTER)
        scls_one_hot_tiled = torch.unsqueeze(scls_one_hot, 2).repeat(1, 1, 3)
        predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*scls_one_hot_tiled, dim=1)

        mean_size_arr_expand = torch.unsqueeze(g_mean_size_arr, 0)
        mean_size_label = torch.sum(scls_one_hot_tiled * mean_size_arr_expand, dim=1)
        size_residual_label_normalized = size_residual_label / mean_size_label
        size_residual_normalized_loss = fn.smooth_l1_loss(size_residual_label_normalized, predicted_size_residual_normalized)

        corners_3d = self.get_box3d_corners(end_points['center'], end_points['heading_residuals'],
                                                                                end_points['size_residuals'])
        gt_mask = torch.unsqueeze(hcls_onehot, 2).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                            torch.unsqueeze(hcls_onehot, 1).repeat(1, NUM_HEADING_BIN, 1)
        corners_3d_pred = torch.sum(torch.sum(torch.unsqueeze(torch.unsqueeze(gt_mask, -1), -1).float() * corners_3d, dim=1), dim=1)

        heading_bin_centers = torch.from_numpy(np.arange(0, 2*np.pi, 2*np.pi/NUM_HEADING_BIN)).float()
        heading_label = torch.unsqueeze(heading_residual_label, 1) + torch.unsqueeze(heading_bin_centers, 0)
        heading_label = torch.sum(hcls_onehot.float() * heading_label, dim=1)

        mean_sizes = torch.unsqueeze(torch.from_numpy(g_mean_size_arr).float(), 0)
        size_label = mean_sizes + torch.unsqueeze(size_residual_label, 1)
        size_label = torch.sum(torch.unsqueeze(scls_one_hot.float(), -1)*size_label, dim=1)
        corners_3d_gt = self.get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = self.get_box3d_corners_helper(center_label, heading_label+np.pi, size_label)

        corners_loss = torch.min(fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt), fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt_flip))

        # Weighted sum of all losses
        loss = mask_loss + box_loss_weight * (center_loss + heading_class_loss + size_class_loss + \
                     heading_residual_normalized_loss*20 + size_residual_normalized_loss*20 + \
                     stage1_center_loss + corner_loss_weight*corners_loss)

        return loss


class CornerLoss_sunrgbd(nn.Module):
    def __init__(self):
        super(CornerLoss_sunrgbd, self).__init__()

    def forward(self, logits, mask_label, center_label, heading_class_label,
                heading_residual_label, size_class_label, size_residual_label,
                end_points, reg_weight=0.001):
        """ logits: BxNxC,
            mask_label: BxN,
            center_label: Bx3
            heading_class_label: B
            heading_residual_label: B
            size_class_label: B
            size_residual_label: Bx3
        """
        B = logits.size(0)
        N = logits.size(1)

        assert logits.size(2) == 2

        assert mask_label.size(0) == B
        assert mask_label.size(1) == N

        assert center_label.size(0) == B
        assert center_label.size(1) == 3

        assert heading_class_label.size(0) == B

        assert heading_residual_label.size(0) == B

        assert size_class_label.size(0) == B

        assert size_residual_label.size(0) == B
        assert size_residual_label.size(1) == 3

        # sparse softmax cross entropy is used
        mask_loss = fn.cross_entropy(torch.transpose(logits, 1, 2), mask_label.long())

        # Huber Loss equivalent is smooth_L1_loss
        center_loss = fn.smooth_l1_loss(end_points['center'], center_label)

        stage1_center_loss = fn.smooth_l1_loss(end_points['stage1_center'], center_label)

        heading_class_loss = fn.cross_entropy(end_points['heading_scores'], heading_class_label.long())

        hcls_onehot = make_onehot(heading_class_label, NUM_HEADING_BIN) # BxNUM_HEADING_BIN
        assert hcls_onehot.size(0) == B
        assert hcls_onehot.size(1) == NUM_HEADING_BIN

        heading_residual_normalized_label = heading_residual_label / (np.pi/NUM_HEADING_BIN)
        heading_loss_input = torch.sum(end_points['heading_residuals_normalized']*hcls_onehot, dim=1)
        heading_residual_normalized_loss = fn.smooth_l1_loss(heading_loss_input, heading_residual_normalized_label)

        size_class_loss = fn.cross_entropy(end_points['size_scores'], size_class_label.long())

        scls_one_hot = make_onehot(size_class_label, NUM_SIZE_CLUSTER) # BxNUM_SIZE_CLUSTER
        assert scls_one_hot.size(0) == B
        assert scls_one_hot.size(1) == NUM_SIZE_CLUSTER

        scls_one_hot_tiled = torch.unsqueeze(scls_one_hot, 2).repeat(1, 1, 3) # BxNUM_SIZE_CLUSTERx3
        assert scls_one_hot_tiled.size(0) == B
        assert scls_one_hot_tiled.size(1) == NUM_SIZE_CLUSTER
        assert scls_one_hot_tiled.size(2) == 3

        predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*scls_one_hot_tiled, dim=1) # Bx3
        assert predicted_size_residual_normalized.size(0) == B
        assert predicted_size_residual_normalized.size(1) == 3

        mean_size_arr_expand = torch.unsqueeze(mean_size_arr, 0).float() # 1xNUM_SIZE_CLUSTERx3
        assert mean_size_arr_expand.size(0) == 1
        assert mean_size_arr_expand.size(1) == NUM_SIZE_CLUSTER
        assert mean_size_arr_expand.size(2) == 3

        mean_size_label = torch.sum(scls_one_hot_tiled * mean_size_arr_expand, dim=1) # Bx3
        assert mean_size_label.size(0) == B
        assert mean_size_label.size(1) == 3

        size_residual_label_normalized = size_residual_label / mean_size_label
        size_residual_normalized_loss = fn.smooth_l1_loss(predicted_size_residual_normalized, size_residual_label_normalized)

        # TODO: Have to add this in computational graph
        # Compute IOU 3D
        iou2ds, iou3ds = self.compute_box3d_iou(end_points['center'], end_points['heading_scores'], end_points['heading_residuals'],
                                                end_points['size_scores'], end_points['size_residuals'], center_label, heading_class_label,
                                                heading_residual_label, size_class_label, size_residual_label)
        end_points['iou2ds'] = iou2ds
        end_points['iou3ds'] = iou3ds

        # Compute BOX3D corners
        corners_3d = get_box3d_corners(end_points['center'], end_points['heading_residuals'],
                                       end_points['size_residuals']) # (B, NH, NS, 8, 3)
        assert corners_3d.size(0) == B
        assert corners_3d.size(1) == NUM_HEADING_BIN
        assert corners_3d.size(2) == NUM_SIZE_CLUSTER
        assert corners_3d.size(3) == 8
        assert corners_3d.size(4) == 3

        gt_mask = torch.unsqueeze(hcls_onehot, 2).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                  torch.unsqueeze(scls_one_hot, 1).repeat(1, NUM_HEADING_BIN, 1) # (B, NH, NS)
        assert gt_mask.size(0) == B
        assert gt_mask.size(1) == NUM_HEADING_BIN
        assert gt_mask.size(2) == NUM_SIZE_CLUSTER

        corners_3d_pred = torch.sum(torch.sum(torch.unsqueeze(torch.unsqueeze(gt_mask, -1), -1).float() * corners_3d, dim=1),dim=1) # (B, 8, 3)
        assert corners_3d_pred.size(0) == B
        assert corners_3d_pred.size(1) == 8
        assert corners_3d_pred.size(2) == 3

        heading_bin_centers = torch.from_numpy(np.arange(0, 2*np.pi, 2*np.pi/NUM_HEADING_BIN)).float() # (NH, )
        assert heading_bin_centers.size(0) == NUM_HEADING_BIN

        heading_label = torch.unsqueeze(heading_residual_label, 1) + torch.unsqueeze(heading_bin_centers, 0).cuda() # (B, NH)
        assert heading_label.size(0) == B
        assert heading_label.size(1) == NUM_HEADING_BIN

        heading_label = torch.sum(hcls_onehot.float() * heading_label, dim=1)

        mean_sizes = torch.unsqueeze(mean_size_arr.float(), 0) # (1, NS, 3)
        size_label = mean_sizes + torch.unsqueeze(size_residual_label, 1) # (1, NS, 3) + (B, 1, 3) = (B, NS, 3)
        size_label = torch.sum(torch.unsqueeze(scls_one_hot.float(), -1)*size_label, dim=1) # (B, 3)
        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label) # (B, 8, 3)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label+np.pi, size_label) # (B, 8, 3)
        assert corners_3d_gt.size(0) == B and corners_3d_gt_flip.size(0) == B
        assert corners_3d_gt.size(1) == 8 and corners_3d_gt_flip.size(1) == 8
        assert corners_3d_gt.size(2) == 3 and corners_3d_gt_flip.size(2) == 3

        corners_loss = torch.min(fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt), fn.smooth_l1_loss(corners_3d_pred, corners_3d_gt_flip))

        return mask_loss + (center_loss + heading_class_loss + size_class_loss + heading_residual_normalized_loss*20 \
                            + size_residual_normalized_loss*20 + stage1_center_loss)*0.1 + corners_loss

    def compute_box3d_iou(self, center_pred, heading_logits, heading_residuals, size_logits, size_residuals,
                          center_label, heading_class_label, heading_residual_label, size_class_label, size_residual_label):
        ''' Used for confidence score supervision..
        Inputs:
            center_pred: (B,3)
            heading_logits: (B,NUM_HEADING_BIN)
            heading_residuals: (B,NUM_HEADING_BIN)
            size_logits: (B,NUM_SIZE_CLUSTER)
            size_residuals: (B,NUM_SIZE_CLUSTER,3)
            center_label: (B,3)
            heading_class_label: (B,)
            heading_residual_label: (B,)
            size_class_label: (B,)
            size_residual_label: (B,3)
        Output:
            iou2ds: (B,) birdeye view oriented 2d box ious
            iou3ds: (B,) 3d box ious
        '''
        batch_size = heading_logits.shape[0]
        heading_class = torch.argmax(heading_logits, 1) # B
        heading_residual = torch.tensor([heading_residuals[i, heading_class[i]] for i in range(batch_size)]).cuda() # B,
        size_class = torch.argmax(size_logits, 1) # B
        size_residual = torch.stack([size_residuals[i, size_class[i],:] for i in range(batch_size)], dim=0)

        iou2d_list = []
        iou3d_list = []
        for i in range(batch_size):
            heading_angle = self.class2angle(heading_class[i], heading_residual[i], NUM_HEADING_BIN)
            box_size = self.class2size(size_class[i], size_residual[i])
            corners_3d = self.get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = self.class2angle(heading_class_label[i], heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = self.class2size(size_class_label[i], size_residual_label[i])
            corners_3d_label = self.get_3d_box(box_size_label, heading_angle_label, center_label[i])

            iou_3d, iou_2d = self.box3d_iou(corners_3d, corners_3d_label)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
        return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)

    def class2angle(self, pred_cls, residual, num_class, to_label_format=True):
        ''' Inverse function to angle2class '''
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center.float().cuda() + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def class2size(self, pred_cls, residual):
        mean_size = glb.type_mean_size[glb.class2type[pred_cls.item()]]
        return torch.from_numpy(mean_size).float().cuda() + residual.cuda()

    def get_3d_box(self, box_size, heading_angle, center):
        ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
            output (8,3) array for 3D box cornders
            Similar to utils/compute_orientation_3d
        '''
        R = torch.from_numpy(self.roty(heading_angle)).float()
        l, w, h = box_size
        x_corners = torch.stack([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=0);
        y_corners = torch.stack([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=0);
        z_corners = torch.stack([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=0);
        corners_3d = torch.matmul(R.cuda(), torch.stack([x_corners,y_corners,z_corners], dim=0).cuda())
        corners_3d[0,:] = corners_3d[0,:] + center[0];
        corners_3d[1,:] = corners_3d[1,:] + center[1];
        corners_3d[2,:] = corners_3d[2,:] + center[2];
        corners_3d = corners_3d.t()
        return corners_3d

    def box3d_iou(self, corners1, corners2):
       ''' Compute 3D bounding box IoU.

       Input:
         corners1: numpy array (8,3), assume up direction is negative Y
         corners2: numpy array (8,3), assume up direction is negative Y
       Output:
         iou: 3D bounding box IoU
         iou_2d: bird's eye view 2D bounding box IoU
      '''
       # corner points are in counter clockwise order
       rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
       rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
       area1 = self.poly_area(torch.stack(rect1[:][0]), torch.stack(rect1[:][1]))
       area2 = self.poly_area(torch.stack(rect2[:][0]), torch.stack(rect2[:][1]))
       inter, inter_area = self.convex_hull_intersection(rect1, rect2)
       iou_2d = inter_area/(area1+area2-inter_area)
       ymax = torch.min(corners1[0,1], corners2[0,1])
       ymin = torch.max(corners1[4,1], corners2[4,1])
       inter_vol = torch.tensor([inter_area]).cuda() * torch.max(torch.tensor([0.0]).cuda(), ymax-ymin)
       vol1 = self.box3d_vol(corners1).cuda()
       vol2 = self.box3d_vol(corners2).cuda()
       iou = inter_vol / (vol1 + vol2 - inter_vol)
       return iou, iou_2d

    def box3d_vol(self, corners):
        ''' corners: (8,3) no assumption on axis direction '''
        a = torch.sqrt(torch.sum((corners[0,:] - corners[1,:])**2))
        b = torch.sqrt(torch.sum((corners[1,:] - corners[2,:])**2))
        c = torch.sqrt(torch.sum((corners[0,:] - corners[4,:])**2))
        return a*b*c

    def roty(self, t):
       """Rotation about the y-axis."""
       c = np.cos(t)
       s = np.sin(t)
       return np.array([[c,  0,  s],
                        [0,  1,  0],
                        [-s, 0,  c]])


    def polygon_clip(self, subjectPolygon, clipPolygon):
        """ Clip a polygon with another polygon.

        Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

        Args:
            subjectPolygon: a list of (x,y) 2d points, any polygon.
            clipPolygon: a list of (x,y) 2d points, has to be *convex*
        Note:
            **points have to be counter-clockwise ordered**

        Return:
            a list of (x,y) vertex point for the intersection polygon.
        """
        def inside(p):
            return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])

        def computeIntersection():
            dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
            dp = [ s[0] - e[0], s[1] - e[1] ]
            n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
            n2 = s[0] * e[1] - s[1] * e[0]
            n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
            return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]

        outputList = subjectPolygon
        cp1 = clipPolygon[-1]

        for clipVertex in clipPolygon:
            cp2 = clipVertex
            inputList = outputList
            outputList = []
            s = inputList[-1]

            for subjectVertex in inputList:
                e = subjectVertex
                if inside(e):
                    if not inside(s):
                        outputList.append(computeIntersection())
                    outputList.append(e)
                elif inside(s):
                    outputList.append(computeIntersection())
                s = e
            cp1 = cp2
            if len(outputList) == 0:
                return None
        return(outputList)

    def convex_hull_intersection(self, p1, p2):
        """ Compute area of two convex hull's intersection area.
            p1,p2 are a list of (x,y) tuples of hull vertices.
            return a list of (x,y) for the intersection and its volume
        """
        inter_p = self.polygon_clip(p1,p2)
        if inter_p is not None:
            hull_inter = ConvexHull(inter_p)
            return inter_p, hull_inter.volume
        else:
            return None, 0.0

    def poly_area(self, x, y):
        """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
        #return 0.5*torch.abs(torch.dot(x, np.roll(y,1)) - torch.dot(y, np.roll(x,1)))
        return 0.5*torch.abs(torch.dot(x, torch.stack([y[-1], y[0]])) - torch.dot(y, torch.stack([x[-1], x[0]])))
