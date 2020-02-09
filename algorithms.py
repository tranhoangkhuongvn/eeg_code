from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from sklearn.metrics import precision_score


use_cuda = True 

device = torch.device(
    "cuda" if use_cuda and torch.cuda.is_available() and 'GeForce' not in torch.cuda.get_device_name(0) else "cpu")
BATCH_SIZE = 100

torch.manual_seed(200)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(1.5, 0.5)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

    if isinstance(m, nn.Linear) and m.out_features != 1:
        nn.init.uniform_(m.weight, -0.5, 0.5)
        nn.init.constant_(m.bias, 1)

class ClassificationNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(ClassificationNetwork, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_size, int(input_size * 3)),
                                     nn.Linear(int(input_size * 3), int(input_size * 3)),
                                     nn.Linear(int(input_size * 3), int(input_size * 2)),
                                     nn.Linear(int(input_size * 2), int(input_size)),
                                     nn.Linear(int(input_size), int(input_size / 3)),
                                     nn.Linear(int(input_size / 3), output_size)
                                     )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adamax(self.parameters(), lr=1e-3)

    def forward(self, x):
        import pdb
        pdb.set_trace()
        return self.network(x)
    
class CNNclassification(nn.Module):

    def __init__(self):
        super(CNNclassification, self).__init__()
#        self.conv1 = nn.Conv2d(1, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 13 * 378, 120)
#        self.fc2 = nn.Linear(120, 120)
#        self.fc3 = nn.Linear(120, 120)
#        self.fc4 = nn.Linear(120, 84)
#        self.fc5 = nn.Linear(84, 4)
         # Layer 1
        self.c1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.b1 = nn.BatchNorm2d(16, False)
        self.p1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.c2 = nn.Conv2d(16, 4, (2, 32))
        self.b2 = nn.BatchNorm2d(4, False)
        self.p2 = nn.MaxPool2d(2, 4)

        self.zp2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.c3 = nn.Conv2d(4, 4, (8, 4))
        self.b3 = nn.BatchNorm2d(4, False)
        self.p3 = nn.MaxPool2d((2, 4))
        self.fc1 = nn.Linear(4*8*91, 1000)
        self.fc_temp = nn.Linear(1000, 4)
        self.fc2 = nn.Linear(1000, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 4)
#
        self.apply(init_weights)
        self.criterion = nn.CrossEntropyLoss()
#        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.01)
#        self.optimizer = optim.Adamax(self.parameters(), lr=1e-5, weight_decay=0.01)
#        self.optimizer = optim.SGD(self.parameters(), lr=1e-5, weight_decay=0.01)

    def forward(self, x):
        x =x.to(device)
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(x.shape[0], 16 * 13 * 378)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
#        x = F.relu(self.fc4(x))
#        x = self.fc5(x)
        x = F.dropout(self.b1(F.elu(self.c1(x))), 0.3)
        x = self.p2(F.dropout(self.b2(F.elu(self.c2(self.p1(x)))), 0.3))
        x = self.p3(F.dropout(self.b3(F.elu(self.c3(self.zp2(x)))), 0.3))

#        x = self.p3(self.b3(self.c3(self.zp2(self.p2(self.b2(F.relu(self.c2(self.p1(self.b1(F.elu(self.c1(x))))))))))))
        x = x.view(-1, 4*8*91)
#        x = self.fc6(torch.sigmoid(self.fc5(torch.sigmoid(self.fc4(torch.sigmoid(self.fc3(self.fc2(self.fc1(x)))))))))
#        x = self.fc6(self.fc5(self.fc4(self.fc3(self.fc2(self.fc1(x))))))
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
#        x = self.fc_temp(self.fc1(x))
        return x

class EEG_Net(nn.Module):
	def __init__(self):
		super(EEG_Net, self).__init__()
		# convolutional layer (sees 1x64x1525 input data)
		self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
		self.batch_norm1 = nn.BatchNorm2d(16) 
		self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
		self.batch_norm2 = nn.BatchNorm2d(32) 
		# max pooling layer
		self.pool = nn.MaxPool2d(2, 2)
		
		self.fc1 = nn.Linear(32 * 16 * 381, 4000)
		
		self.fc2 = nn.Linear(4000, 2000)

		self.fc3 = nn.Linear(2000, 4)
		# dropout layer (p=0.25)
		self.dropout = nn.Dropout(0.25)

	def forward(self, x):
		# add sequence of convolutional and max pooling layers
		x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
		x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
		# flatten image input
		x = x.view(-1, 32 * 16 * 381)
		# add dropout layer
		x = self.dropout(x)
		# add 1st hidden layer, with relu activation function
		x = F.relu(self.fc1(x))
		# add dropout layer
		x = self.dropout(x)
		# add 2nd hidden layer, with relu activation function
		x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)

		return x

