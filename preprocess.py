import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym

""" modified from:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
"""

class RepeatActionAndMaxFrame(gym.Wrapper):

    def __init__(self, env=None, repeat=4, clip_reward=False,
                 no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        #self.frame_buffer = np.zeros_like((2,self.shape))
        # Change feb 28, 10:26
        self.frame_buffer = np.zeros(shape=(2, *self.shape), dtype=np.uint8)
        self.clip_reward = clip_reward
        self.no_ops = 0
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        #self.frame_buffer = np.zeros_like((2,self.shape))
        # Change feb 28, 10:26
        self.frame_buffer = np.zeros(shape=(2, *self.shape), dtype=np.uint8)
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0,
                                              shape=self.shape,dtype=np.float32)
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(repeat, axis=0),
                             env.observation_space.high.repeat(repeat, axis=0),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env


'''
import gym
import cv2
import numpy as np
from collections import deque


class RepeatActionAndMaxFrame(gym.wrappers):
    def __init__(self, numOperationsToSkip, fireFirst, env=None, numFrames=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)

        self.numOperationsToSkip = numOperationsToSkip
        self.fireFirst = fireFirst
        self.shape = env.obeservation_space.low.shape
        self.numFrames = numFrames
        self.frameBuffer = np.zeros_like((2, self.shape))

    def step(self, action):
        #Overloaded step function performs the provided action on the environment numFrames times
        #and computes the totalReward and the maxFrame. It returns the environment
        #response in typical openai gym format: (maxFrame, totalReward, isDone, info)
        
        totalReward = 0.0
        isDone = False
        for i in range(self.numFrames):
            (observation, reward, isDone, info) = self.env.step(action)
            totalReward += reward
            self.frameBuffer[i % 2] = observation
            if isDone:
                break

        # Find the element-wise maximum of array elements
        maxFrame = np.maximum(self.frameBuffer[0], self.frameBuffer[1])

        return maxFrame, totalReward, isDone, info

    def reset(self):
        #Overloaded reset function, resets the environment and reinitialize the frame buffer.
        
        initialObservation = self.env.reset()

        if self.numOperationsToSkip > 0:
            self.numOperationsToSkip = np.random.randint(self.numOperationsToSkip) + 1
        else:
            self.numOperationsToSkip = 0

        # Ignore numOperationsToSkip observations
        for i in range(self.numOperationsToSkip):
            (_, _, isDone, _) = self.env.step(0)
            if isDone:
                self.env.reset()

        if self.fireFirst:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            (obs, _, _, _) = self.env.step(1)

        self.frameBuffer = np.zeros_like((2, self.shape))
        self.frameBuffer[0] = initialObservation

        return initialObservation


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(PreprocessFrame, self).__init__(env)

        # Move channel from last position to first position, necessary for PyTorch
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0,high=1.0,shape=self.shape,dtype=np.float32)

    def observation(self, observation):
        #Overloaded function which converts the observation frame to gray scale and
        #resizes it.

        # Convert frame to gray scale
        grayFrame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # Resize the frame
        resizedScreen = cv2.resize(grayFrame,
                                   self.shape[1:],
                                   interpolation=cv2.INTER_AREA)

        # Reshape and convert to array
        resizedObservation = np.array(resizedScreen, dtype=np.uint8).reshape(self.shape)
        resizedObservation = resizedObservation / 255

        return resizedObservation


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        # Repeat low and high bounds repeat number of times
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = deque(maxlen=repeat)

    def reset(self):
        #Overloaded reset function resets the  frame stack,
        #resets the environment and repeats the initial observation
        #reset times. The function also converts the stack into numpy
        #array and returns it.

        self.stack.clear()

        initialObservation = self.env.reset()
        for i in range(self.stack.maxlen):
            self.stack.append(initialObservation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, obs):
        #Overloaded observation function, which appends observation to the frame stack
        #reshapes it and returns it.

        self.stack.append(obs)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs


def makeEnv(envName, newShape=(84,84,1), repeat=4, numOperationsToSkip=0, fireFirst=False):
    # Function takes care of creating a custom environment

    # Make the base environment
    env = gym.make(envName)

    # Add functionalities to the base environment
    env = RepeatActionAndMaxFrame(numOperationsToSkip, fireFirst, env)
    env = PreprocessFrame(newShape, env)
    env = StackFrames(env, repeat)

    return env


'''


