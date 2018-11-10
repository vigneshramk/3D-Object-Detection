import sys
import math
import numpy as np
import torch
import torch.nn as nn
import os
import models.Mother as Mother
from data.sunrgbd_loader import SUN_TrainDataSet,SUN_TrainLoader
from loss import CornerLoss_sunrgbd
import models.globalVariables as glb
from hyperParams import hyp
from logger import logger

os.environ["CUDA_VISIBLE_DEVICES"]= hyp["gpu"]
use_cuda = torch.cuda.is_available()
print('Cuda')


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
        self.iou_2d_per_class = tensor.zeros(10)          # (B, ) -- No. of Classes
        self.iou_3d_per_class = tensor.zeros(10)          # (B, ) -- No. of Classes
        self.valid_loss = []
        self.metrics = {}
        self.logger=logger()
        self.log=self.logger.log
        self.plot = self.logger.plot        #array,label,epoch or array,label
        self.log("Hyperparameters : {}".format(hyp))

        self.log_dir = 'log_directory/' + hyp["log_dir"]
        # Create the results directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.model_dir = '../results' + hyp["test_name"]

        # Create the results directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)


    def save_checkpoint(self):
        save_dict = {
            "iou_2d": self.iou_2d_per_class.numpy(),
            "iou_3d": self.iou_3d_per_class.numpy()
        }
        torch.save(save_dict, file_save)    # Saves train params


    def load_checkpoint(self, fname_model, fname_hyp = None):
        load_dict = torch.load(fname_model)
        self.model.load_state_dict(load_dict['model_state_dict'])
        if (fname_hyp is not None):
            hyp = np.load(fname_hyp)[()]    # Loads dictionary from npy file
            self.log("Hyperparameters loaded from checkpoint as {}".format(hyp))


    def eval(self, val_loader):
        self.model.eval()
        lossfn = CornerLoss_sunrgbd()
        self.log("Start Evaluation...")
        class_count = np.zeros(10)
        for batch_num, (features, class_labels, labels_dict) in enumerate(val_loader):
            X = torch.FloatTensor(features).requires_grad_()
            X = X.cuda()
            class_labels_one_hot = one_hot_encoding(class_labels)
            Y = torch.FloatTensor(class_labels_one_hot)
            Y = Y.cuda()

            logits, end_points = self.model(X, Y)

            for key in labels_dict.keys():
                labels_dict[key] = labels_dict[key].cuda()


            iou2ds, iou3ds = lossfn.compute_box3d_iou(end_points['center'], end_points['heading_scores'], end_points['heading_residuals'],
                                                      end_points['size_scores'], end_points['size_residuals'], labels_dict['center_label'],
                                                      labels_dict['heading_class_label'], labels_dict['heading_residual_label'],
                                                      labels_dict['size_class_label'], labels_dict['size_residual_label'])

            # Storing iou2ds and iou3ds
            for i, label in enumerate(class_labels):
                iou_2d_per_class[label] += iou2ds[i].item()
                iou_3d_per_class[label] += iou3ds[i].item()
                class_count[label] += 1

        for i in range(10):
            iou_2d_per_class[i] = iou_2d_per_class[i]/float(class_count[i])
            iou_3d_per_class[i] = iou_3d_per_class[i]/float(class_count[i])

        # Saves entire history of train loss over batches & valid loss over epoch
        self.save_checkpoint()
        self.logger.close()


# Runs as a script when called
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Need Model File.')

    # Instantiate models
    net = Mother.Model()
    model_trainer = Eval(net)
    model_trainer.load_checkpoint(sys.argv[1])
    train_dataset = SUN_TrainDataSet(2048)
    val_loader = SUN_TrainLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=True,num_workers=hyp["num_workers"], pin_memory=False)
    model_trainer.eval(val_loader)

