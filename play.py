import gym
from gym.utils import play
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-env", type=str, default="PongNoFrameskip-v4", help="Atari environment.")
    parser.add_argument("-fps",type=int,default=120,help="Frames per second.")
    args = parser.parse_args()

    play.play(gym.make(args.env), zoom=5,fps=args.fps)