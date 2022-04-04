import os
import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, numActions, inputDim, learnRate, savePath, networkName):
        super(DQN, self).__init__()

        # Save network name and path
        self.networkName = networkName
        self.savePath = os.path.join(savePath, networkName)

        # Convolutional layers
        # nn.Conv2d(in_channels, filters, kernel_size, stride)
        self.conv1 = nn.Conv2d(inputDim[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # Fully connected layers
        fcInputDimension = self.__calculateConvOutputDimension(inputDim)
        self.fc1 = nn.Linear(fcInputDimension, 512)
        self.fc2 = nn.Linear(512, numActions)

        # Define the optimizer
        self.optimizer = optim.RMSprop(self.parameters(), lr=learnRate)

        # Define the loss function (Mean Squared Error loss)
        self.loss = nn.MSELoss()

        # Find and send network to the available device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        print("device = ", self.device)

    def __calculateConvOutputDimension(self, inputDim):
        state = T.zeros(1, *inputDim)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, stateList):
        '''
        Function passes a batch (size 1 or more) of states
        through the network and returns a set of action values.
        actions = [Q(s,a) for each action a]
        '''

        conv1 = F.relu(self.conv1(stateList))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions

    def save(self):
        '''
        Function saves the neural network to savePath.
        '''
        print('Saving: ' + self.networkName + " ...")
        T.save(self.state_dict(), self.savePath)

    def load(self):
        '''
        Function loads the neural network from savePath.
        '''
        print('Loading: ' + self.networkName + " ...")
        self.load_state_dict(T.load(self.savePath))
