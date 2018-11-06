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
from tensorboardX import SummaryWriter

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

# Training cradle
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model.cuda()
        if hyp["parallel"]:
            self.model = nn.DataParallel(self.model)
        self.optimizer = optimizer
        self.epoch = 0
        self.train_batch_loss = []
        self.train_epoch_loss = []
        self.valid_loss = []
        self.metrics = {}
        self.logger=logger()
        self.log=self.logger.log
        self.plot = self.logger.plot        #array,label,epoch or array,label
        self.log("Hyperparameters : {}".format(hyp))

        self.log_dir = 'log_directory/' + hyp["log_dir"]
        self.writer = SummaryWriter(self.log_dir)
        
        # Create the results directory
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.model_dir = '../results' + hyp["test_name"]
        
        # Create the results directory
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def save_checkpoint(self):
        save_dict = {
        "epoch_idx": self.epoch + 1, 
        "model_state_dict": self.model.state_dict(), 
        "optim_state_dict":self.optimizer.state_dict(), 
        #"training_loss":self.train_avg_loss,
        "training_epoch_loss": self.train_epoch_loss,
        # "val_loss":self.valid_loss
        }

        fname_model = hyp["model_fname"]+"ep"+str(self.epoch)+".pth"
        fname_hyp = hyp["hyp_fname"]+"ep"+str(self.epoch)+".txt"

        file_save = self.model_dir + '/' + fname_model
        np_save = self.model_dir + '/' + fname_hyp

        torch.save(save_dict, file_save)    # Saves train params
        np.save(np_save, hyp)               # Saves hyperparams dictionary    

    def load_checkpoint(self, fname_model, fname_hyp = None):
        load_dict = torch.load(fname_model)
        self.epoch = load_dict['epoch_idx']
        self.train_epoch_loss = load_dict['training_epoch_loss']
        # self.valid_loss = load_dict['val_loss']
        self.model.load_state_dict(load_dict['model_state_dict'])
        self.optimizer.load_state_dict(load_dict['optim_state_dict'])

        if (fname_hyp is not None):
            hyp = np.load(fname_hyp)[()]    # Loads dictionary from npy file
            self.log("Hyperparameters loaded from checkpoint as {}".format(hyp))

    def run(self, train_loader, val_loader, epochs=hyp["num_epochs"]):
        self.model.train()
        #torch.autograd.set_detect_anomaly(True)
        lossfn = CornerLoss_sunrgbd()
        self.log("Start Training...")
        niter = 0
        for epoch in range(epochs):
            self.train_batch_loss = []
            valid_batch_loss = 0  # Resets valid loss for each epoch

            for batch_num, (features, class_labels, labels_dict) in enumerate(train_loader):
                self.optimizer.zero_grad()
                X = torch.FloatTensor(features).requires_grad_()
                X = X.cuda()
                class_labels = one_hot_encoding(class_labels)
                Y = torch.FloatTensor(class_labels)
                Y = Y.cuda()
                     
                logits, end_points = self.model(X, Y)

                for key in labels_dict.keys():
                    labels_dict[key]=labels_dict[key].cuda()

                corner_loss = lossfn(logits, labels_dict['mask_label'], labels_dict['center_label'], 
                                labels_dict['heading_class_label'], labels_dict['heading_residual_label'], 
                                labels_dict['size_class_label'], labels_dict['size_residual_label'], end_points)

                if batch_num % hyp["log_freq"] ==0:
                    self.log("Batch number: {0}, loss: {1:.6f}".format(batch_num+1, corner_loss.item()))


                # Checks for loss exploding
                if math.isnan(corner_loss.item()):
                  #  self.save_checkpoint("fault.pth","fault.txt")
                    for key in end_points.keys():
                        try:
                            if torch.isnan(end_points[key]).any():
                                self.log("Loss exploded @{}. Dumped:{}".format(key,end_points[key]))
                        except:
                            pass
                    if torch.isnan(logits).any():
                        self.log("Loss exploded @logits. Dumped:{}".format(logits))
                    sys.exit("Loss exploded!")

                # Implements gradient clipping if desired
                if (hyp["grad_clip"]):
                    nn.utils.clip_grad_value_(self.model.parameters(), hyp["grad_clip"])

                corner_loss.backward()
                
                self.optimizer.step()
                
                # Keeps track of batch loss and running mean of batch losses
                self.train_batch_loss.append(corner_loss.item())

                self.writer.add_scalar('data/iter_loss',corner_loss.item(),niter)
                niter +=1

            # Stores last entry in running average of batch losses as epoch loss
            self.train_epoch_loss.append(np.mean(self.train_batch_loss))
            self.writer.add_scalar('data/epoch_loss',np.mean(self.train_batch_loss),epoch)

            #print("Training for %d epoch completed", %epoch)
            
            if False:

                self.model.eval()
                #torch.autograd.set_detect_anomaly(False)
                for batch_idx, (val_features, val_class_labels, val_labels_dict) in enumerate(val_loader):
                    X_val = torch.FloatTensor(val_features)
                    X_val = X_val.cuda()

                    val_class_labels = one_hot_encoding(val_class_labels)
                    Y_val = torch.FloatTensor(val_class_labels)
                    Y_val = Y_val.cuda()

                    val_logits, val_end_points = self.model(X_val, Y_val)

                    for key in val_labels_dict.keys():
                        val_labels_dict[key]=val_labels_dict[key].cuda()

                    # May want to sum losses and average in a way acc. to number of points rather than simple averaging ?
                    valid_batch_loss += lossfn(val_logits, val_labels_dict['mask_label'], val_labels_dict['center_label'], 
                                            val_labels_dict['heading_class_label'], val_labels_dict['heading_residual_label'], 
                                            val_labels_dict['size_class_label'], val_labels_dict['size_residual_label'], val_end_points)
                # Averages valid loss
                self.valid_loss.append(valid_batch_loss/(batch_idx+1))

            self.metrics["train_loss_{}".format(epoch)] = self.train_epoch_loss[-1]
            # self.metrics["valid_loss_{}".format(epoch)] = self.valid_loss[-1]
            # Saves entire history of train loss over batches & valid loss over epoch
            self.save_checkpoint()

            #self.log("epoch:", epoch+1, "train avg loss:", round(self.train_epoch_loss[-1],4))
        self.logger.close()


# # Runs as a script when called
# if __name__ == "__main__":
# Instantiate models
net = Mother.Model()
AdamOptimizer = torch.optim.Adam(net.parameters(), lr=hyp['lr'], weight_decay=hyp['optim_reg'])

model_trainer = Trainer(net, AdamOptimizer)
train_dataset = SUN_TrainDataSet(2048)
train_loader = SUN_TrainLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=True,num_workers=hyp["num_workers"], pin_memory=False)
val_loader = SUN_TrainLoader(train_dataset, batch_size=hyp["batch_size"], shuffle=True,num_workers=hyp["num_workers"], pin_memory=False)
model_trainer.run(train_loader, val_loader, epochs=hyp['num_epochs'])

