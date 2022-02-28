import numpy as np


class ReplayMemoryBuffer():
    def __init__(self, maxSize, inputShape, numActions):

        self.size = maxSize
        self.indexPointer = 0

        # Init containers
        self.states = np.zeros((self.size, *inputShape), dtype=np.float32)
        self.newStates = np.zeros((self.size, *inputShape), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.isDones = np.zeros(self.size, dtype=np.bool)
        # self.isDones = np.zeros(self.size, dtype=np.bool_)

    def storeTransition(self, state, action, reward, isDone, newState):
        '''
        Function stores transitions (state,action,reward, is done flag,next state)
        in the first unoccupied position.
        If memory is full position is selected by the modulo operator.
        '''

        transitionIndex = self.indexPointer % self.size

        self.states[transitionIndex] = state
        self.newStates[transitionIndex] = newState
        self.actions[transitionIndex] = action
        self.rewards[transitionIndex] = reward
        self.isDones[transitionIndex] = isDone

        self.indexPointer += 1

    def sampleTransitions(self, sampleSize):
        '''
        Function uniformly samples transitions with no duplicates
        and returns a tuple of transitions.
        '''

        maxMem = min(self.indexPointer, self.size)
        randomBatch = np.random.choice(maxMem, sampleSize, replace=False) # False => No repeating.

        states = self.states[randomBatch]
        newStates = self.newStates[randomBatch]
        actions = self.actions[randomBatch]
        rewards = self.rewards[randomBatch]
        isDones = self.isDones[randomBatch]

        return states, newStates, actions, rewards, isDones








