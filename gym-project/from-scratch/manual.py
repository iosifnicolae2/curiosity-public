import argparse
import random
import time
import datetime

import numpy
import torch
import torch_ac
import tensorboardX
import sys

import keyboard

import gym
import gym_minigrid

import matplotlib.pyplot as plt


def get_env():
    env_key = 'MiniGrid-FourRooms-v0'
    seed=None
    env = gym.make(env_key)
    env.seed(seed)
    return env

def start_the_game():
    env = get_env()
    current_obs = env.reset()
    env.render()
    while True:

        action = None
        # for ac in [1, 2, 3, 4, 5, 6, 7, ]:
        #     if keyboard.is_pressed(str(ac)):
        #         action = int(ac)

        action = random.choice([0, 1, 2, 3, 4, 5, 6])
        if not action:
            time.sleep(0.1)
            continue

        obs, reward, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    start_the_game()