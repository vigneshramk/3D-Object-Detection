Dataloader Part 1 - Pickle file representation in loader.py


id_list,box2d_list,box3d_list,input_list,label_list,type_list,heading_list,box3d_size_list,frustum_angle_list=load_zipped_pickle('sunrgbd_3d.pickle')

Pickle format

1. id_list - Image ids - <id>.jpg files in the img folder of mysunrgbd data (integer with 6 digits. e.g 000066 for id 66)

2. box2d_list - 2D bounding box coords of the form [x1,y1,x2,y2]

3. box3d_list - (8,3) array in upright depth coord with 8 corners for each of x,y,z

4. input_list - frustum input channel= 6, xyz,rgb in upright depth coord
MAIN variable containing the (nxc) FRUSTUM. 
NOTE: here c=6 i.e. xyz, rgb. To convert to c=4, just average the rgb values (might be sub-optimal). So do, intensity = 0.299 R + 0.587 G + 0.114 B 
(got c=6 so that we can use rgb channels separately if required)

5. label_list 

6. type_list - Class type (str) e.g. sofa. Use type to class to get the class label (and convert into one-hot vector) 
NOTE: type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

7. heading_list - face of object angle, radius of clockwise angle from positive x axis in upright camera coord

8. box3d_size_list - array of l,w,h for the 3d box

9. frustum_angle_list - angle of 2d box center from pos x-axis



Dataloader Part 2 - Output of the batching function in sunrgbd_loader.py
image_id_batch,frustum_batch,class_batch,labels_dict

1. image_id_batch - list of len batch_size containing the id of the data file used  (integer with 6 digits. e.g 000066 for id 66)

2. frustum_batch - tensor (batch_size,2048,4) - containing the xyz, intensity frustum input 

3. class_batch - tensor (batch_size,) containing the class label, int between 0-9

4. labels_dict - contains all the labels in the dict keys as follows

	4.1 labels_dict['mask_label'] - (batch_size,2048) containing 0 for clutter and 1 for roi
    4.2 labels_dict['center_label'] - (batch_size,3) containing the center for the 3D box
    
    4.3,4.4 labels_dict['heading_class_label'], labels_dict['heading_residual_label'] - intTensor (batch_size,) and floatTensor (batch_size,) converting the continuous angle to discrete with class and residual
    
    4.5,4.6 labels_dict['size_class_label'] labels_dict['size_residual_label'] - - intTensor (batch_size,) and floatTensor (batch_size,3) Convert 3D box size (l,w,h) to size class and size residual

    4.7 labels_dict['rotate_angle'] - FloatTensor (batch_size,) center view rotation angle of the frustum