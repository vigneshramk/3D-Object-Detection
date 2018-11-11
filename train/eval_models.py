import sys
import math
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', linewidth=2)

import models.Mother as Mother
from data.sunrgbd_loader import SUN_TrainDataSet,SUN_TrainLoader
from loss import CornerLoss_sunrgbd
import models.globalVariables as glb
from hyperParams import hyp
from eval_det import eval_det

os.environ["CUDA_VISIBLE_DEVICES"]= hyp["gpu"]
use_cuda = torch.cuda.is_available()
print('Cuda')
classname_list = ['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']

# Function for transforming interger labels to one-hot vectors
def one_hot_encoding(class_labels, num_classes = glb.NUM_CLASS):
    # # Dictionary for converting string labels into one hot vectors
    # class_dict={'bed':0, 'table':1, 'sofa':2, 'chair':3, 'toilet':4, 'desk':5, 'dresser':6, 'night_stand':7, 'bookshelf':8, 'bathtub':9}

    num_inputs = class_labels.shape[0]
    labels_one_hot = np.zeros((num_inputs, num_classes))
    for i in range(num_inputs):
        labels_one_hot[i][class_labels[i]] = 1
    return labels_one_hot


class Eval:
    def __init__(self, model):
        self.model = model.cuda()
        if hyp["parallel"]:
            self.model = nn.DataParallel(self.model)
        self.epoch = 0
        self.iou_2d_per_class = torch.zeros(glb.NUM_CLASS)          # (B, ) -- No. of Classes
        self.iou_3d_per_class = torch.zeros(glb.NUM_CLASS)          # (B, ) -- No. of Classes
        self.valid_loss = []
        self.metrics = {}

        self.log_dir = 'log_directory/' + hyp["log_dir"]
        # Create the results directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_dir = '../results' + hyp["test_name"]

        # Create the results directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


    def load_checkpoint(self, fname_model, fname_hyp = None):
        load_dict = torch.load(fname_model)
        self.model.load_state_dict(load_dict['model_state_dict'])
        if (fname_hyp is not None):
            hyp = np.load(fname_hyp)[()]    # Loads dictionary from npy file


    def eval(self, val_loader):
        gt_all = {}
        pred_all = {}
        ovthresh = 0.25

        self.model.eval()
        lossfn = CornerLoss_sunrgbd(evaluate=True)
        class_count = np.zeros(glb.NUM_CLASS)
        for batch_num, (img_id, features, class_labels, labels_dict) in enumerate(val_loader):
            X = torch.FloatTensor(features).requires_grad_()
            X = X.cuda()
            class_labels_one_hot = one_hot_encoding(class_labels)
            Y = torch.FloatTensor(class_labels_one_hot)
            Y = Y.cuda()

            logits, end_points = self.model(X, Y)
            for key in labels_dict.keys():
                labels_dict[key] = labels_dict[key].cuda()

            iou2ds, iou3ds, corners_3d_gt, corners_3d_pred = lossfn(logits, labels_dict['mask_label'], labels_dict['center_label'],
                                                                    labels_dict['heading_class_label'], labels_dict['heading_residual_label'],
                                                                    labels_dict['size_class_label'], labels_dict['size_residual_label'], end_points)

            corners_3d_gt = corners_3d_gt.cpu().numpy()
            corners_3d_pred = corners_3d_pred.cpu().numpy()
            scores = end_points['size_scores'].cpu().numpy()
            # Storing iou2ds and iou3ds
            for i, label in enumerate(class_labels):
                self.iou_2d_per_class[label.item()] += iou2ds[i].item()
                self.iou_3d_per_class[label.item()] += iou3ds[i].item()
                class_count[label.item()] += 1
                if img_id[i] not in gt_all:
                  gt_all[img_id[i]] = []
                  pred_all[img_id[i]] = []
                gt_all[img_id[i]].append((classname_list[label.item()], corners_3d_gt[i]))
                pred_all[img_id[i]].append((classname_list[label.item()], corners_3d_pred[i], scores[i]))

        for i in range(10):
            self.iou_2d_per_class[i] = self.iou_2d_per_class[i]/float(class_count[i])
            self.iou_3d_per_class[i] = self.iou_3d_per_class[i]/float(class_count[i])

        print('Computing AP...')
        rec, prec, ap = eval_det(pred_all, gt_all, ovthresh)
        for classname in ap.keys():
            print('%015s: %f' % (classname, ap[classname]))
            plt.plot(rec[classname], prec[classname], lw=3)
            fig = plt.gcf()
            fig.subplots_adjust(bottom=0.25)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=24)
            plt.ylabel('Precision', fontsize=24)
            plt.title(classname, fontsize=24)
            plt.savefig(self.model_dir + '/' + classname + '.png')
            plt.close()
        print('mean AP: ', np.mean([ap[classname] for classname in ap]))


# Runs as a script when called
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Need Model File.')

    # Instantiate models
    net = Mother.Model()
    model_trainer = Eval(net)
    model_trainer.load_checkpoint(sys.argv[1])
    train_dataset = SUN_TrainDataSet(2048)
    val_loader = SUN_TrainLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=False, num_workers=hyp["num_workers"], pin_memory=False)
    with torch.no_grad():
        model_trainer.eval(val_loader)
