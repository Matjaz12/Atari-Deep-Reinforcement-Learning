## Pong using Deep Reinforcement Learning
### Info
Deep Q Learning Agent trained on the Pong game. \
Agent beats the best human player with just a few hours of \
training using a personal computer.

The project was developed for course: \
`Artificial Intelligence (Master's degree, FRI 2022)`  

### Requirements

`pip install torch` \
`pip install torchvision` \
`pip install numpy` \
`pip install matplotlib` \
`pip install gym` \
`pip install box2d-py` \
`pip install atari-py` \
`pip install pygame` \
`pip install cv2` \
`pip install ale-py`

### Usage
Evaluate pong agent & plot distribution of actions \
`python -u main.py -memSize=50000 -mode=eval -loadModel=1 -path=models/ -actionHist=1 -numEpisodes=5 -env=PongNoFrameskip-v4` \
Evaluate freeway agent & plot ditribution of actions \
`python -u main.py -memSize=50000 -mode=eval -loadModel=1 -path=models/ -actionHist=1 -numEpisodes=5 -algo=DDQN -env=FreewayNoFrameskip-v4\
Train DDQN Agent on freeway \
`python -u main.py -memSize=50000 -path=models/ -algo=DDQN -env=FreewayNoFrameskip-v4`
### Documentation
Brief explanation of the project is available in `paper.pdf`.


