from agent import Agent
from preprocess import make_env
from _plot import *

NUM_EPISODES = 500
NETWORK_NAME = "DQN"
EVALUATION = "Base"
PLOTS_PATH = "plots/" + NETWORK_NAME + "_" + EVALUATION

# https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with
# Demo game play
# https://www.udemy.com/course/deep-q-learning-from-paper-to-code/learn/lecture/17009498#questions/10488006
if __name__ == "__main__":
    # Make environment
    env = make_env('PongNoFrameskip-v4')

    # Initialize agent
    agent = Agent(discountFactor=0.99,
                  epsilonMax=1,
                  epsilonMin=0.1,
                  epsilonDec=1e-5,
                  learnRate=0.0001,
                  inputDim=(env.observation_space.shape),
                  numActions=env.action_space.n,
                  replayMemoryBufferSize=50000, # 50000 => takes about 13GB of RAM!
                  replayMemoryBatchSize=32,
                  targetNetworkUpdateInterval=1000,
                  networkSavePath="models/",
                  evaluationName = "base",
                  networkName="DQN",
                  trainingPhase=True
                  )

    loadModel = False
    if loadModel:
        agent.loadModel()

    # Main training loop
    bestScore = -np.inf
    stepCounter = 0
    scores, epsilons, steps = [], [], []

    for episodeCounter in range(NUM_EPISODES):
        episodeScore = 0
        observation = env.reset()

        isDone = False
        while not isDone:
            # Select action
            action = agent.selectAction(observation)

            # Execute action in emulator
            (newObservation, reward, isDone, info) = env.step(action)
            episodeScore += reward

            # Learn
            if not loadModel:
                agent.storeTransition(observation, action, reward, isDone, newObservation)
                agent.learn()

            observation = newObservation
            stepCounter += 1

        scores.append(episodeScore)
        steps.append(stepCounter)
        epsilons.append(agent.epsilon)

        # Compute score over the previous 100 number of games
        averageScore = np.mean(scores[-100:])
        print("episode: ", episodeCounter,
              "score: ", episodeScore,
              "average score:", averageScore,
              "best score: ", bestScore,
              "epsilon: ", agent.epsilon,
              "steps: ", stepCounter)

        # Save the model
        if averageScore > bestScore:
            if not loadModel:
                # agent.saveModel()
                pass

            bestScore = averageScore

    # Plot scores as the function of learning steps taken
    agent.saveModel()
    plot_learning_curve(steps, scores, epsilons, filename=PLOTS_PATH)










