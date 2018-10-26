import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    mean_size_arr[i,:] = type_mean_size[class2type[i]]

def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    print '-----', centers
    N = centers.size(0)
    l = sizes[:,0]	 # (N,1) 
    w = sizes[:,1] # (N,1)
    h = sizes[:,2] # (N,1)
    print l,w,h
    x_corners = torch.cat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim=1) # (N,8)
    y_corners = torch.cat([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim=1) # (N,8)
    z_corners = torch.cat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1) # (N,8)
    corners = torch.cat([x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)], dim=1) # (N,3,8)
    print x_corners, y_corners, z_corners
    c = torch.cos(headings)
    s = torch.sin(headings)
    ones = torch.ones(N)
    zeros = torch.zeros(N)
    row1 = torch.stack([c,zeros,s], dim=1) # (N,3)
    row2 = torch.stack([zeros,ones,zeros], dim=1)
    row3 = torch.stack([-s,zeros,c], dim=1)
    R = torch.cat([row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)], dim=1) # (N,3,3)
    print row1, row2, row3, R, N
    corners_3d = torch.matmul(R, corners) # (N,3,8)
    corners_3d += centers.unsqueeze(2).repeat(1,1,8) # (N,3,8)
    corners_3d = corners_3d.permute(0,2,1) # (N,8,3)
    return corners_3d

def get_box3d_corners(center, heading_residuals, size_residuals):
    """ TF layer.
    Inputs:
        center: (B,3)
        heading_residuals: (B,NH)
        size_residuals: (B,NS,3)
    Outputs:
        box3d_corners: (B,NH,NS,8,3) tensor
    """
    batch_size = center.size(0)
    heading_bin_centers = torch.FloatTensor(np.arange(0,2*np.pi,2*np.pi/NUM_HEADING_BIN)) # (NH,) This was a 32 bit float
    headings = heading_residuals + heading_bin_centers.view(1,heading_bin_centers.size(0)) # (B,NH)
    
    mean_sizes = torch.FloatTensor(mean_size_arr).unsqueeze(0) + size_residuals # (B,NS,1) This was a 32 bit float
    sizes = mean_sizes + size_residuals # (B,NS,3)
    sizes = sizes.unsqueeze(1).repeat(1,NUM_HEADING_BIN,1,1) # (B,NH,NS,3)
    headings = headings.unsqueeze(-1).repeat(1,1,NUM_SIZE_CLUSTER) # (B,NH,NS)
    centers = center.unsqueeze(1).unsqueeze(1).repeat(1,NUM_HEADING_BIN, NUM_SIZE_CLUSTER,1) # (B,NH,NS,3)

    N = batch_size*NUM_HEADING_BIN*NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N,3), headings.view(N), sizes.view(N,3))

    return corners_3d.view(batch_size, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)
