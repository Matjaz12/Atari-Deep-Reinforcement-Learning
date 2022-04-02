import sys
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from preprocess import make_env
from plotting import *
from gym import wrappers
from train import *


# Program Parameters
DQN_AGENT = 1
DDQN_AGENT = 2
PLAY_MODE = 1
TRAIN_MODE = 2
ENVIRONMENT = "PongNoFrameskip-v4"

# Agent Parameters
GAMMA = 0.99
EPS_MAX = 1.00
EPS_MIN = 0.1
EPS_DEC = 1e-5
LEARN_RATE = 0.0001
REPLAY_MEM_SIZE = 50000
REPLAY_SAMPLE_SIZE = 32
TARGET_NET_UPDATE_INTERVAL = 1000
NETWORKS_PATH = "models/"
PLOTS_PATH = "plots/"
NUM_EPISODES = 5000

if __name__ == "__main__":
    agent = int(sys.argv[1])
    mode  = int(sys.argv[2])

    if agent == DQN_AGENT:
        if mode == TRAIN_MODE:
            env = make_env(ENVIRONMENT)
            agent = DQNAgent(gamma=GAMMA,
                             epsilonMax=EPS_MAX,
                             epsilonMin=EPS_MIN,
                             epsilonDec=EPS_DEC,
                             learnRate=LEARN_RATE,
                             inputDim=env.observation_space.shape,
                             numActions=env.action_space.n,
                             replayMemoryBufferSize=REPLAY_MEM_SIZE,
                             replayMemoryBatchSize=REPLAY_SAMPLE_SIZE,
                             targetNetworkUpdateInterval=TARGET_NET_UPDATE_INTERVAL,
                             networkSavePath=NETWORKS_PATH,
                             evaluationName="base",
                             networkName="DQN",
                             trainingPhase=True)

            (scoreList, epsilonList, stepList, valueEstimates) = trainAgent(agent, env, NUM_EPISODES,
                                                                            saveAgent=True, trainMode=True, verbose=True)

            # Plotting
            plotLearnCurve(stepList, scoreList, epsilonList, filename=PLOTS_PATH + "DQN")
            plotValueEstimates(valueEstimates, title="Pong", filename=PLOTS_PATH + "VAL_ESTIMATE_DQN_TRAIN")

        if mode == PLAY_MODE:
            env = make_env(ENVIRONMENT)
            env = wrappers.Monitor(env, "tmp/dqn", video_callable=lambda episode_id: True, force=True)
            agent = DQNAgent(gamma=GAMMA,
                             epsilonMax=EPS_MAX,
                             epsilonMin=EPS_MIN,
                             epsilonDec=EPS_DEC,
                             learnRate=LEARN_RATE,
                             inputDim=env.observation_space.shape,
                             numActions=env.action_space.n,
                             replayMemoryBufferSize=REPLAY_MEM_SIZE,
                             replayMemoryBatchSize=REPLAY_SAMPLE_SIZE,
                             targetNetworkUpdateInterval=TARGET_NET_UPDATE_INTERVAL,
                             networkSavePath=NETWORKS_PATH,
                             evaluationName="base",
                             networkName="DQN",
                             trainingPhase=False)

            agent.loadModel()
            (scoreList, epsilonList, stepList, valueEstimates) = trainAgent(agent, env, NUM_EPISODES,
                                                                            saveAgent=False, trainMode=False, verbose=True)
            # Plotting
            plotValueEstimates(valueEstimates, title="Pong", filename=PLOTS_PATH + "VAL_ESTIMATE_DQN_PLAY")

        else:
            print("Incorrect usage!")
            print("main.py <AGENT> <MODE>")
            print("<DQN_AGENT> = <1>, <DDQN_AGENT> = <2>")
            print("<PLAY_MODE> = <1>, <TRAIN_MODE> = <2>")

    if agent == DDQN_AGENT:
        if mode == TRAIN_MODE:
            env = make_env(ENVIRONMENT)
            agent = DDQNAgent(gamma=GAMMA,
                              epsilonMax=EPS_MAX,
                              epsilonMin=EPS_MIN,
                              epsilonDec=EPS_DEC,
                              learnRate=LEARN_RATE,
                              inputDim=env.observation_space.shape,
                              numActions=env.action_space.n,
                              replayMemoryBufferSize=REPLAY_MEM_SIZE,
                              replayMemoryBatchSize=REPLAY_SAMPLE_SIZE,
                              targetNetworkUpdateInterval=TARGET_NET_UPDATE_INTERVAL,
                              networkSavePath=NETWORKS_PATH,
                              evaluationName="base",
                              networkName="DQN",
                              trainingPhase=True)

            (scoreList, epsilonList, stepList, valueEstimates) = trainAgent(agent, env, NUM_EPISODES,
                                                                            saveAgent=True, trainMode=True, verbose=True)

            # Plotting
            plotLearnCurve(stepList, scoreList, epsilonList, filename=PLOTS_PATH + "DDQN")
            plotValueEstimates(valueEstimates, title="Pong", filename=PLOTS_PATH + "VAL_ESTIMATE_DDQN_TRAIN")

        if mode == PLAY_MODE:
            env = make_env(ENVIRONMENT)
            env = wrappers.Monitor(env, "tmp/dqn", video_callable=lambda episode_id: True, force=True)
            agent = DDQNAgent(gamma=GAMMA,
                              epsilonMax=EPS_MAX,
                              epsilonMin=EPS_MIN,
                              epsilonDec=EPS_DEC,
                              learnRate=LEARN_RATE,
                              inputDim=env.observation_space.shape,
                              numActions=env.action_space.n,
                              replayMemoryBufferSize=REPLAY_MEM_SIZE,
                              replayMemoryBatchSize=REPLAY_SAMPLE_SIZE,
                              targetNetworkUpdateInterval=TARGET_NET_UPDATE_INTERVAL,
                              networkSavePath=NETWORKS_PATH,
                              evaluationName="base",
                              networkName="DQN",
                              trainingPhase=False)

            (scoreList, epsilonList, stepList, valueEstimates) = trainAgent(agent, env, NUM_EPISODES,
                                                                            saveAgent=False, trainMode=False, verbose=True)

            # Plotting
            plotValueEstimates(valueEstimates, title="Pong", filename=PLOTS_PATH + "VAL_ESTIMATE_DDQN_PLAY")

        else:
            print("Incorrect usage!")
            print("main.py <AGENT> <MODE>")
            print("<DQN_AGENT> = <1>, <DDQN_AGENT> = <2>")
            print("<PLAY_MODE> = <1>, <TRAIN_MODE> = <2>")

    else:
        print("Incorrect usage!")
        print("main.py <AGENT> <MODE>")
        print("<DQN_AGENT> = <1>, <DDQN_AGENT> = <2>")
        print("<PLAY_MODE> = <1>, <TRAIN_MODE> = <2>")


