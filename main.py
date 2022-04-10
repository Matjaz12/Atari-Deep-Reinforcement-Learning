from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from preprocess import make_env
from gym import wrappers
from train import *
from datetime import datetime
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-numEpisodes", type=int, default=500, help="Number of episodes to play.")
    parser.add_argument("-learnRate", type=float, default=0.0001, help="Model learning rate.")
    parser.add_argument("-epsMax", type=float, default=1.00, help="Maximum value of epsilon in epsilon greedy method.")
    parser.add_argument("-epsMin", type=float, default=0.1, help="Minimum value of epsilon in epsilon greedy method.")
    parser.add_argument("-epsDec", type=float, default=1e-5, help="Decrement value of epsilon in epsilon greedy method.")
    parser.add_argument("-gamma", type=float, default=0.99, help="Discount factor for update Q equation.")
    parser.add_argument("-memSize", type=int, default=50000, help="Replay memory buffer size.")
    parser.add_argument("-batchSize", type=int, default=32, help="Replay memory sample batch size.")
    parser.add_argument("-replaceInterval", type=int, default=1000, help="Target network replace weights interval.")
    parser.add_argument("-env", type=str, default="PongNoFrameskip-v4", help="Atari environment.")
    parser.add_argument("-loadModel", type=bool, default=False, help="Load model checkpoint.")
    parser.add_argument("-path", type=str, default="models/", help="Path for model saving/loading.")
    parser.add_argument("-evalName", type=str, default=datetime.now().strftime("%H:%M:%S"), help="Model evaluation name")
    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
    parser.add_argument("-algo", type=str, default="DQN", help="DQN/DDQN")
    parser.add_argument("-mode", type=str, default="train", help="train/eval")

    args = parser.parse_args()

    # Display selected arguments
    print("-----------------------------")
    for arg in vars(args):
        print(f"{arg} = {getattr(args, arg)}")
    print("-----------------------------")

    # Enable/Disable GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.algo == "DQN":
        if args.mode == "train":
            env = make_env(args.env)
            agent = DQNAgent(gamma=args.gamma,
                             epsilonMax=args.epsMax,
                             epsilonMin=args.epsMin,
                             epsilonDec=args.epsDec,
                             learnRate=args.learnRate,
                             inputDim=env.observation_space.shape,
                             numActions=env.action_space.n,
                             replayMemoryBufferSize=args.memSize,
                             replayMemoryBatchSize=args.batchSize,
                             targetNetworkUpdateInterval=args.replaceInterval,
                             networkSavePath=args.path,
                             evaluationName="eval",
                             networkName="DQN",
                             trainingPhase=True)

            if args.loadModel:
                agent.loadModel()

            scoreList, epsilonList, stepList = trainAgent(agent, env, args.numEpisodes,
                                                                            saveAgent=True, trainMode=True,verbose=True)

        if args.mode == "eval":
            env = make_env(args.env)
            env = wrappers.Monitor(env, args.path + "tmp/dqn", video_callable=lambda episode_id: True, force=True)

            agent = DQNAgent(gamma=args.gamma,
                             epsilonMax=args.epsMax,
                             epsilonMin=args.epsMin,
                             epsilonDec=args.epsDec,
                             learnRate=args.learnRate,
                             inputDim=env.observation_space.shape,
                             numActions=env.action_space.n,
                             replayMemoryBufferSize=args.memSize,
                             replayMemoryBatchSize=args.batchSize,
                             targetNetworkUpdateInterval=args.replaceInterval,
                             networkSavePath=args.path,
                             evaluationName="base",
                             networkName="DQN",
                             trainingPhase=True)

            agent.loadModel()
            scoreList, epsilonList, stepList = trainAgent(agent, env, args.numEpisodes,
                                                          saveAgent=False, trainMode=False,verbose=True)
    if args.algo == "DDQN":
        if args.mode == "train":
            env = make_env(args.env)
            agent = DDQNAgent(gamma=args.gamma,
                              epsilonMax=args.epsMax,
                              epsilonMin=args.epsMin,
                              epsilonDec=args.epsDec,
                              learnRate=args.learnRate,
                              inputDim=env.observation_space.shape,
                              numActions=env.action_space.n,
                              replayMemoryBufferSize=args.memSize,
                              replayMemoryBatchSize=args.batchSize,
                              targetNetworkUpdateInterval=args.replaceInterval,
                              networkSavePath=args.path,
                              evaluationName="base",
                              networkName="DDQN",
                              trainingPhase=True)

            if args.loadModel:
                agent.loadModel()

            scoreList, epsilonList, stepList = trainAgent(agent, env, args.numEpisodes,
                                                          saveAgent=True, trainMode=True,verbose=True)

        if args.mode == "eval":
            env = make_env(args.env)
            env = wrappers.Monitor(env, args.path + "tmp/ddqn", video_callable=lambda episode_id: True, force=True)
            agent = DDQNAgent(gamma=args.gamma,
                              epsilonMax=args.epsMax,
                              epsilonMin=args.epsMin,
                              epsilonDec=args.epsDec,
                              learnRate=args.learnRate,
                              inputDim=env.observation_space.shape,
                              numActions=env.action_space.n,
                              replayMemoryBufferSize=args.memSize,
                              replayMemoryBatchSize=args.batchSize,
                              targetNetworkUpdateInterval=args.replaceInterval,
                              networkSavePath=args.path,
                              evaluationName="base",
                              networkName="DDQN",
                              trainingPhase=True)

            scoreList, epsilonList, stepList = trainAgent(agent, env, args.numEpisodes,
                                                          saveAgent=False, trainMode=False,verbose=True)