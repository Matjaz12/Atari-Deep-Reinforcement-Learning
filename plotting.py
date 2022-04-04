import matplotlib.pyplot as plt
import numpy as np


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


def plotValueEstimates(valueEstimates, title, filename):
    plt.plot(valueEstimates)
    plt.xlabel("T")
    plt.ylabel("Value Estimate")
    plt.title(title)

    plt.savefig(filename)





