import matplotlib.pyplot as plt
import numpy as np
import os


def plotLearnCurve(episodes, scores, epsilons, filename, runningAvgInterval=100):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    # Plot epsilon as a function of episodes
    ax1.plot(episodes, epsilons, color="cornflowerblue")
    ax1.set_xlabel("training steps")
    ax1.set_ylabel("epsilon", color="cornflowerblue")
    ax1.tick_params(axis="x", colors="black")
    ax1.tick_params(axis="y", colors="black")

    runningAverageScores = [np.mean(scores[max(0, t - runningAvgInterval): (t + 1)]) for t in range(len(scores))]
    ax2.plot(episodes, runningAverageScores, color="red")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel("score", color="red")
    ax2.yaxis.set_label_position("right")
    ax1.tick_params(axis="y")

    plt.tight_layout()
    plt.savefig(filename)


def plotLearnCurveFromLogs(folder, fileName):
    episodes, scores, epsilons = [], [], []

    with open(os.path.join(folder, fileName + ".log"), 'r', encoding='utf8') as f:
        f.readline()
        for line in f:
            lineList = line.split(" ")
            if "episode=" in lineList[2]:
                for i in range(3, len(lineList)):
                    param = lineList[i].split("=")
                    name, value = param
                    if name == "step":
                        episodes.append(int(value))
                    elif name == "score":
                        scores.append(float(value))
                    elif name == "epsilon":
                        epsilons.append(float(value))

    plotLearnCurve(episodes, scores, epsilons, "plots/" + fileName)


def plotLearnCurvesFromLogs(folder, listOfFiles):
    _episodes, _scores, _epsilons = [], [], []

    for file in listOfFiles:
        episodes, scores, epsilons = [], [], []
        with open(os.path.join(folder, file + ".log"), 'r', encoding='utf8') as f:
            f.readline()
            for line in f:
                lineList = line.split(" ")
                if "episode=" in lineList[0]:
                    for i in range(1, len(lineList)):
                        param = lineList[i].split("=")
                        name, value = param
                        if name == "step":
                            episodes.append(int(value))
                        elif name == "score":
                            scores.append(float(value))
                        elif name == "epsilon":
                            epsilons.append(float(value))

        _episodes.append(episodes)
        _scores.appends(scores)
        _epsilons.append(epsilons)

    # TODO: implement plotting of multiple functions.


def plotActionHistogram(actionHist, envName):
    totalActions = sum([val for key, val in actionHist.items()])
    for key, val in actionHist.items():
        actionHist[key] = val / totalActions

    if "Pong" in envName:
        ########################################################################
        # Pong specific (multiple actions actually do the same thing!)
        actionHistCorrected = dict()
        actionHistCorrected[0] = 0
        actionHistCorrected[1] = 0
        actionHistCorrected[2] = 0

        for key, val in actionHist.items():
            if key == 0 or key == 1:
                actionHistCorrected[0] += val
            elif key == 2 or key == 4:
                actionHistCorrected[1] += val
            elif key == 3 or key == 5:
                actionHistCorrected[2] += val

        plt.bar(list(actionHistCorrected.keys()), actionHistCorrected.values())
        ########################################################################
    else:
        plt.bar(list(actionHist.keys()), actionHist.values())

    plt.title(f"Distribution of actions over {totalActions} steps")
    plt.ylabel("Frequency")
    plt.xlabel('Action')
    plt.show()


if __name__ == "__main__":
    plotLearnCurveFromLogs(folder="./logs",fileName="freeway_dqn")