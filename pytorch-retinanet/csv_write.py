import csv
import os
import utils_sun

type2class={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

# root_dir = '/home/vignesh/Projects/3D-Object-Detection/2d_data'
root_dir = '/home/kvr/Documents/Projects/3D-Object-Detection/2d_data/training'

image_dir = os.path.join(root_dir, 'image')
label_dir = os.path.join(root_dir, 'label_dimension')


idx = 0
with open('sun_csv/annotate.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for root, dirs, files in os.walk(image_dir):
        for file in files:

            name = file.strip('.jpg')
            img_path = image_dir + '/' + file
            label_filename = os.path.join(label_dir, '%s.txt'%(name))
            objects = utils_sun.read_sunrgbd_label(label_filename)

            for o in objects:
                obj = o.__dict__
                cl = obj['classname']
                if cl not in type2class:
                    continue
                x1,y1,x2,y2 = map(int,obj['box2d'])
                if x2 <= x1 or y2 <= y1:
                    continue
                writer.writerow([img_path,x1,y1,x2,y2,cl])


with open('sun_csv/map.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for obj,ind in type2class.items():
        writer.writerow([obj,ind])

            


    
