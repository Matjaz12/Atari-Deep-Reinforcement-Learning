import numpy as np


class ReplayMemoryBuffer():
    def __init__(self, maxSize, inputShape, numActions):

        self.size = maxSize
        self.indexPointer = 0

        # Init containers
        # self.size ... number of rows
        # *inputShape ... number of columns => size of observation space
        self.states = np.zeros((self.size, *inputShape), dtype=np.float32)
        self.newStates = np.zeros((self.size, *inputShape), dtype=np.float32)

        # since these are just scalars: rows x cols => 1 x self.size
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
        # maxMem => [0, 1, ... maxMem) : bucket from which we sample
        # sampleSize => size of the sample (how many times we pull from above bucket)
        # e.g np.random.choise(5,3) => e.g (4,0,2)
        randomBatch = np.random.choice(maxMem, sampleSize, replace=False)  # False => No repeating.

        # get values at index-es in randomBatch
        states = self.states[randomBatch]
        newStates = self.newStates[randomBatch]
        actions = self.actions[randomBatch]
        rewards = self.rewards[randomBatch]
        isDones = self.isDones[randomBatch]

        return states, newStates, actions, rewards, isDones








