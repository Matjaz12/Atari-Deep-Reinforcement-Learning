import random


class RANDOMAgent:
    def __init__(self, numActions, networkName, computeActionHist=False):
        # Store parameters
        self.numActions = numActions
        self.actionSpace = [i for i in range(numActions)]
        self.networkName = networkName
        self.computeActionHist = computeActionHist

        # Create action histogram
        if self.computeActionHist:
            self.actionHist = dict()
            for i in range(0, numActions):
                self.actionHist[i] = 0

    def selectAction(self, observation):
        action = random.choice(self.actionSpace)

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





