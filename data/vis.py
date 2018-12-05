import numpy as np
from loader import load_zipped_pickle
from viz_util import draw_lidar_simple
import mayavi.mlab as mlab











data=load_zipped_pickle("vis_sunrgbd.pickle")
pc,id_list,box2d_list,box3d_list,input_list,label_list,type_list,heading_list,box3d_size_list,frustum_angle_list = [np.array(elem) for elem in data]
pc = [np.array(i[:,:3]) for i in pc]
fig = draw_lidar_simple(pc[0])
mlab.savefig('pc_view.jpg', figure=fig)
input()
