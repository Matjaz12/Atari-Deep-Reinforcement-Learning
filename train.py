import logging
from datetime import datetime
import numpy as np


def trainAgent(agent, env, numEpisodes, saveAgent=True, trainMode=True, log=False):
    bestScore = -np.inf
    stepCounter = 0
    scoreList, epsilonList, stepList = [], [], []

    if log:
        logging.basicConfig(filename="logs/" + agent.networkName + datetime.now().strftime("%H:%M:%S") + ".log",
                            format='%(asctime)s %(message)s',
                            filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

    # Iterate over each episode
    for episodeCounter in range(numEpisodes):
        # Init episode score get initial observation
        episodeScore = 0
        observation = env.reset()

        # Play until game over.
        isDone = False
        while not isDone:

            # Select action
            action = agent.selectAction(observation)

            # Execute action in emulator
            (newObservation, reward, isDone, info) = env.step(action)

            # Update episode score
            episodeScore += reward

            if trainMode:
                # Store current transition
                agent.storeTransition(state=observation,
                                      action=action,
                                      reward=reward,
                                      isDone=isDone,
                                      newState=newObservation)
                # Train the agent
                agent.learn()

            # Update observation
            observation = newObservation
            stepCounter += 1

        scoreList.append(episodeScore)
        # stepList.append(stepCounter)
        # epsilonList.append(agent.epsilon)

        # Compute score over the previous 100 number of games
        averageScore = np.mean(scoreList[-100:])

        if log:
            logger.info(f"episode={episodeCounter} score={episodeScore} avgScore={averageScore},"
                        f"bestScore={bestScore} epsilon={agent.epsilon}, step={stepCounter}")

        print(f"episode={episodeCounter} score={episodeScore} avgScore={averageScore},"
                        f"bestScore={bestScore} epsilon={agent.epsilon}, step={stepCounter}")

        if averageScore > bestScore:
            bestScore = averageScore
            if saveAgent:
                agent.saveModel()

    agent.saveModel()

    return scoreList, epsilonList, stepList




