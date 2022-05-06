import random


class RANDOMAgent:
    def __init__(self, numActions, networkName, computeActionHist=False):
        # Store parameters
        self.numActions = numActions
        self.actionSpace = [i for i in range(numActions)]
        self.networkName = networkName
        self.computeActionHist = computeActionHist
<<<<<<< HEAD
=======
        self.epsilon = 1.0
>>>>>>> 1152f6aa1aab1ab42f3b4e5df4ee6f3bbf8ba1d3

        # Create action histogram
        if self.computeActionHist:
            self.actionHist = dict()
            for i in range(0, numActions):
                self.actionHist[i] = 0

    def selectAction(self, observation):
<<<<<<< HEAD
        action = random.choice(self.actionSpace)

=======
        #action = random.choice(self.actionSpace)
        action = 1
>>>>>>> 1152f6aa1aab1ab42f3b4e5df4ee6f3bbf8ba1d3
        if self.computeActionHist:
            self.actionHist[action] += 1

        return action

    def learn(self):
        pass

    def storeTransition(self, state, action, reward, isDone, newState):
        pass

    def saveModel(self):
        print("Im " + self.networkName + ", nothing to save..")
        pass

    def loadModel(self):
        print("Im " + self.networkName + ", nothing to load..")
        pass





