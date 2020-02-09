from sklearn.metrics import accuracy_score
import os
import sys
import time
import pickle
from progress.spinner import Spinner
sys.path.append('./')
import numpy as np
import torch
from torch import nn
from data_utils import EEGDATA
from algorithms import CNNclassification, ClassificationNetwork, EEG_Net
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as torch_data_utils
import torch.optim as optim

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

BATCH_SIZE = 20
n_fold = 5

torch.manual_seed(400)
USE_SAVED_DATA = True
SAVE_DATA_FILE = 'data.pik'


if __name__ == '__main__':
	eeg_data = EEGDATA()
	if (not USE_SAVED_DATA) or (not os.path.exists(SAVE_DATA_FILE)):
		data = eeg_data.load()

		# reshape the data such that input to the network is data of all 64 channels and all time frames
		neural_net_input_data = []
		labels = []
		for class_label, class_data in enumerate(data.values()):
			neural_net_input_data.append(class_data.reshape(eeg_data.num_of_channels*eeg_data.num_of_time_frames, -1))
			# +1 because I dont like zeros
			labels += [class_label for i in range(class_data.shape[-1])]
		
		input_data = []
		for i in neural_net_input_data:
			for item in i.T:
				input_data.append(item)

		input_data = torch.FloatTensor(input_data)
		labels = torch.FloatTensor(labels)
		
		with open(SAVE_DATA_FILE, 'wb') as io:
			pickle.dump({'input_data': input_data, 'labels': labels}, io)

	if USE_SAVED_DATA:
		with open(SAVE_DATA_FILE, 'rb') as io:
			k = pickle.load(io)
			input_data = k['input_data']
			labels = k['labels']
	print('input_data.shape:', input_data.shape)
	print('labels:', labels.shape)
	#labels = [l - 1 for l in labels]
	labels = labels - 1.0
	print(labels)

	valid_size = 0.2
	batch_size = 20
	# obtain training indices that will be used for validation
	num_train = input_data.shape[0] 
	indices = list(range(num_train))
	np.random.shuffle(indices)
	split = int(np.floor(valid_size * num_train))
	train_idx, valid_idx = indices[split:], indices[:split]
	train_data = torch_data_utils.TensorDataset(input_data, labels)
	# define samplers for obtaining training and validation batches
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)


	# prepare data loaders (combine dataset and sampler)
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)
	valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0)

	print(train_loader)
	for i in range(10):
		print(train_data[-i])	
	
	model = EEG_Net()	
	print(model)
	# move tensors to GPU if CUDA is available
	if train_on_gpu:
		model.cuda()


	# specify loss function
	criterion = nn.CrossEntropyLoss()

	# specify optimizer
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	# number of epochs to train the model
	n_epochs = 20 # you may increase this number to train a final model

	valid_loss_min = np.Inf # track change in validation loss

	for epoch in range(1, n_epochs+1):

		# keep track of training and validation loss
		train_loss = 0.0
		valid_loss = 0.0
		
		###################
		# train the model #
		###################
		model.train()
		for data, target in train_loader:
			# move tensors to GPU if CUDA is available
			if train_on_gpu:
				data, target = data.cuda(), target.long().cuda()
			# clear the gradients of all optimized variables


			optimizer.zero_grad()
			# forward pass: compute predicted outputs by passing inputs to the model
			data = data.reshape(BATCH_SIZE, 1, eeg_data.num_of_channels, eeg_data.num_of_time_frames)
			output = model(data)
			# calculate the batch loss
			print(output.dtype, target.dtype)
			loss = criterion(output, target.long())
			# backward pass: compute gradient of the loss with respect to model parameters
			loss.backward()
			# perform a single optimization step (parameter update)
			optimizer.step()
			# update training loss
			train_loss += loss.item()*data.size(0)
			
		######################    
		# validate the model #
		######################
		model.eval()
		for data, target in valid_loader:
			# move tensors to GPU if CUDA is available
			if train_on_gpu:
				data, target = data.cuda(), target.cuda()
			# forward pass: compute predicted outputs by passing inputs to the model
			output = model(data)
			# calculate the batch loss
			loss = criterion(output, target)
			# update average validation loss 
			valid_loss += loss.item()*data.size(0)
		
		# calculate average losses
		train_loss = train_loss/len(train_loader.dataset)
		valid_loss = valid_loss/len(valid_loader.dataset)
			
		# print training/validation statistics 
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
			epoch, train_loss, valid_loss))
		
		# save model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
			valid_loss_min,
			valid_loss))
			torch.save(model.state_dict(), 'model_eeg.pt')
			valid_loss_min = valid_loss
	'''


	neural_net = CNNclassification()
	neural_net = neural_net.to(device)
	indicies = list(np.arange(len(input_data)))
	np.random.shuffle(indicies)
	input_data = np.take(input_data, indicies, 0)
	labels = np.take(labels, indicies)
	input_batch = [batch.reshape(batch.shape[0], 1, eeg_data.num_of_channels, eeg_data.num_of_time_frames) for batch in torch.split(input_data, BATCH_SIZE)]
	label_batch = torch.split(labels, BATCH_SIZE)
	print("create batch")
	for _x in range(n_fold):
		input_train = input_batch[:_x] + input_batch[_x+1:]
		input_label = label_batch[:_x]+ label_batch[_x+1:]
		test_expected = label_batch[_x]
		####check dimension####
		print((len(input_train), len(input_train[0])))
		print(input_train[0])


		print('learning fold {}'.format(_x))
		test_x = input_batch[_x]
		test_y = test_expected
		for i in range(20):
			batch = torch.cat(input_train)
			k = torch.cat(input_label)-1
			k = torch.split(k, BATCH_SIZE)
			for j in range(2000):
				spinner = Spinner('Running epoch {}'.format(j))
				for _temp, _batch in enumerate(torch.split(batch, BATCH_SIZE)):
					for _o in range(30):
						actual = neural_net(_batch)
						print('actual:', actual.shape)
						print('k:', k[_temp].shape)
						exit(0)
						loss = neural_net.criterion(actual, k[_temp].long().to(device))
						neural_net.optimizer.zero_grad()
						loss.backward()
						neural_net.optimizer.step()
						spinner.next()
				if j%10==0 and j != 0:
					_temp = neural_net(test_x)
					pred_test = torch.argmax(_temp, 1)
					print(_temp)
					print(pred_test)
					print(test_y)
					pred_test = pred_test.to(device)
					accuracy = np.round(accuracy_score(pred_test.detach().cpu().numpy(), test_y-1)*100, 3)
					print('Test accuracy {}'.format(accuracy))
					accuracy = []
					for b, _b in enumerate(torch.split(batch, BATCH_SIZE)):
						_temp = neural_net(_b)
						pred_test = torch.argmax(_temp, 1)
						accuracy.append(np.round(accuracy_score(pred_test.detach().cpu().numpy(), k[b])*100, 3))
					print('Train accuracy {}'.format(np.mean(accuracy)))

			torch.save(neural_net.state_dict(), 'workload_model_{}'.format(i))
	
	'''
