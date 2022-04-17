import random
from dqn import DQN
from replay_memory_buffer import ReplayMemoryBuffer
import torch as T
import numpy as np


class DQNAgent:
    def __init__(self, numActions, inputDim,
                 learnRate, epsilonMax, epsilonMin, epsilonDec, gamma,
                 replayMemoryBufferSize, replayMemoryBatchSize, targetNetworkUpdateInterval,
                 networkSavePath, networkName, evaluationName, computeActionHist=False):

        # Save agent parameters
        self.numActions = numActions
        self.inputDim = inputDim
        # action space defined in OpenAI gym format
        self.actionSpace = [i for i in range(numActions)]
        self.learnRate = learnRate
        self.epsilon = epsilonMax
        self.epsilonMin = epsilonMin
        self.epsilonDec = epsilonDec
        self.gamma = gamma
        self.replayMemoryBatchSize = replayMemoryBatchSize
        self.targetNetworkUpdateInterval = targetNetworkUpdateInterval
        self.learnIterations = 0
        self.networkName = networkName

        # Initialize replay memory buffer
        self.replayMemoryBuffer = ReplayMemoryBuffer(maxSize=replayMemoryBufferSize,
                                                     numActions=numActions,
                                                     inputShape=inputDim)

        # Initialize two neural networks
        self.evaluationDQN = DQN(numActions=self.numActions, inputDim=self.inputDim, learnRate=self.learnRate,
                                 networkName=networkName + "_eval_" + evaluationName, savePath=networkSavePath)

        self.targetDQN = DQN(numActions=self.numActions, inputDim=self.inputDim, learnRate=self.learnRate,
                             networkName=networkName + "_target_" + evaluationName, savePath=networkSavePath)

        self.computeActionHist = computeActionHist
        # Create action histogram
        if self.computeActionHist:
            self.actionHist = dict()
            for i in range(0, numActions):
                self.actionHist[i] = 0

    def selectAction(self, observation):
        '''
        Function takes the observation of the current state of environment as input and
        returns either a random action (note that this happens only if Agent is in trainingPhase
        or a greedy (predicted) action.
        '''

        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            # Convert observation to PyTorch tensor before it is loaded to the device.
            # Note that since DQN (which is a convolutional neural network under the hood)
            # takes a tensor of shape batchSize x inputDim, we add a extra dimension by
            # putting a single observation in a list and converting it to a tensor.
            state = T.tensor([observation], dtype=T.float).to(self.evaluationDQN.device)

            # Evaluate Q(state, action) for each action
            actions = self.evaluationDQN.forward(state)

            # Pick action with max Q(state, action).
            action = T.argmax(actions).item()

            if self.computeActionHist:
                self.actionHist[action] += 1

        return action

    def learn(self):
        # Check if we have enough transition saved before we start learning
        if self.replayMemoryBuffer.indexPointer < self.replayMemoryBatchSize:
            return

        # Zero out the gradients from previous iteration
        self.evaluationDQN.optimizer.zero_grad()

        # Attempt to update the target network
        self.__updateTargetDQN()

        # Sample random mini batch of transitions
        (states, newStates, actions, rewards, isDones) = self.__sampleTransitions()

        # Compute predicted Q values
        indices = np.arange(self.replayMemoryBatchSize)  # indices = [0, 1, 3, ..., replayMemoryBatchSize - 1]

        # Pass states from mini batch through network & get predicted q values
        # for the set of actions in the mini batch
        predictedQ = self.evaluationDQN.forward(states)[indices, actions]

        # Compute target Q values
        # self.targetDQN.forward(newStates).max(dim=1)[0] => predicted max q values for all newStates
        nextQ = self.targetDQN.forward(newStates).max(dim=1)[0]  # .max() => [0] ... values,  [1] ... indices

        # Set nextQ to zero, for newStates that terminated.
        nextQ[isDones] = 0.0

        # Reward for performing actions plus discounted estimate of future reward
        targetQ = rewards + self.gamma * nextQ

        # Compute loss
        loss = self.evaluationDQN.loss(targetQ, predictedQ).to(self.evaluationDQN.device)

        # Alter the weights of the neural network using computed loss
        loss.backward()
        self.evaluationDQN.optimizer.step()

        # Decrement the epsilon value
        self.__decrementEpsilon()

        # Up the lear iteration counter
        self.learnIterations += 1

    def storeTransition(self, state, action, reward, isDone, newState):
        self.replayMemoryBuffer.storeTransition(state, action, reward, isDone, newState)

    def __sampleTransitions(self):
        (states, newStates, actions, rewards, isDones) = \
            self.replayMemoryBuffer.sampleTransitions(self.replayMemoryBatchSize)

        # Convert to PyTorch tensor
        statesT = T.tensor(states).to(self.evaluationDQN.device)
        newStatesT = T.tensor(newStates).to(self.evaluationDQN.device)
        actionsT = T.tensor(actions).to(self.evaluationDQN.device)
        rewardsT = T.tensor(rewards).to(self.evaluationDQN.device)
        isDonesT = T.tensor(isDones).to(self.evaluationDQN.device)

        return statesT, newStatesT, actionsT, rewardsT, isDonesT

    def __updateTargetDQN(self):
        if self.learnIterations % self.targetNetworkUpdateInterval == 0:
            # Every targetNetworkUpdateInterval times update the target dqn.
            # We do this by updating the state dictionary of target dqn
            # using the state dictionary of evaluation dqn
            self.targetDQN.load_state_dict(self.evaluationDQN.state_dict())

    def __decrementEpsilon(self):
        if self.epsilon > self.epsilonMin:
            self.epsilon = self.epsilon - self.epsilonDec

    def saveModel(self):
        self.evaluationDQN.save()
        self.targetDQN.save()

    def loadModel(self):
        self.evaluationDQN.load()
        self.targetDQN.load()







