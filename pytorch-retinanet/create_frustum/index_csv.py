import csv
import os
# import utils_sun

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

root_dir = '/home/vignesh/Projects/3D-Object-Detection/2d_data'
image_dir = os.path.join(root_dir, 'image')
label_dir = os.path.join(root_dir, 'label_dimension')


idx = 1
with open('csv_files/train_list.txt', 'w') as file:
    while idx < 10335:
        index = str(idx).zfill(6)
        file.write(index + "\n")
        idx += 1


            


    
