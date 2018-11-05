import numpy as np
import torch
import Mother
import loss
import globalVariables as glb
from hyperParams import hyp

# Instantiate models
net = Mother.Model()

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
		self.model = model
		self.optimizer = optimizer
		self.epoch = 0
		self.train_batch_loss = []
		self.train_avg_loss = []
		self.train_epoch_loss = []
		self.valid_loss = []
		self.metrics = {}
            
	def save_checkpoint(self, train_loss, valid_loss, fname_model = "Train_v1.pth", fname_hyp = "Hyp_v1.pth"):
		save_dict = {"epoch_idx": self.epoch + 1, "model_state_dict": self.model.state_state(), 
		"optim_state_dict":self.optimizer.state_dict(), "training_loss":train_loss, "Val_loss":valid_loss}
		torch.save(save_dict, fname_model)
		torch.save(hyp, fname_hyp)

	def load_checkpoint(self, fname_model, fname_hyp = None):
		load_dict = torch.load(fname_model)
		self.model.load_state_dict(load_dict['model_state_dict'])
		self.optimizer.load_state_dict(load_dict['optim_state_dict'])

		if (fname_hyp is not None):
			hyp = torch.load(fname_hyp)

	def run(self, train_loader, val_loader, epochs=num_epochs):
		print("Start Training...")
		for epoch in range(epochs):
			for batch_num, (features, class_labels, labels_dict) in enumerate(train_loader):
				self.optimizer.zero_grad()
				X = torch.FloatTensor(features).requires_grad_()
				X = X.cuda()
				class_labels = one_hot_encoding(class_labels)
				Y = torch.FloatTensor(class_labels.astype(np.float32))
				Y = Y.cuda()

				logits, end_points = self.model(X, Y)
				
				# labels_dict = mask_label, center_label, heading_class_label, heading_residual_label, 
				# size_class_label, size_residual_label, end_points
				corner_loss = loss(logits, labels_dict['mask_label'], labels_dict['center_label'], heading_class_label, heading_residual_label, 
								size_class_label, size_residual_label, end_points)
				corner_loss.backward()
				
				self.optimizer.step()
				
				self.train_batch_loss.append(corner_loss.item())
				self.train_avg_loss.append(np.mean(self.train_batch_loss))
				
				if batch_num % 100 ==0:
					print("Gradient update: {0}, loss: {1:.8f}".format(print_counter+1, corner_loss.item()))
			
			self.train_epoch_loss.append(self.train_avg_loss[-1])

			valid_loss = 0	# Resets valid loss for each epoch
			for batch_idx, (val_features, val_class_labels, val_labels_dict) in enumerate(val_loader):
				X_val = torch.FloatTensor(val_features)
				X_val = X_val.cuda()

				val_class_labels = one_hot_encoding(val_class_labels)
				Y_val = torch.FloatTensor(val_class_labels.astype(np.float32))
				Y_val = Y_val.cuda()

				logits, end_points = self.model(X_val, Y_val)
				valid_loss += loss(logits, labels_dict['mask_label'], labels_dict['center_label'], heading_class_label, heading_residual_label, 
								size_class_label, size_residual_label, end_points)
			
			self.valid_loss.append(valid_loss/batch_idx)

			self.metrics["valid_loss_{}".format(epoch)] = self.valid_loss[-1]
			self.metrics["train_loss_{}".format(epoch)] = self.train_epoch_loss[-1]
			save_checkpoint(self, self.train_avg_loss, self.valid_loss[-1])


			print("epoch:", epoch+1, "loss:", round(total_loss,4), "train acc.:", round((1.0 - train_error), 4))
			print("Validation set EER:", val_eer, "threshold", val_thresh)
			self.metrics.append(Metric(loss=total_loss, train_error=train_error, val_eer=val_eer))
			self.save_model('./model_basic'+str(epoch+1)+'.pt')