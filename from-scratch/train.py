#!/usr/bin/env python3

import numpy
import yaml

import matplotlib.pyplot as plt

from getkey import getkey, keys
from gym_unity.envs import UnityEnv


class Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, current_observation):

        return self.env.action_space.sample()

        # wait for a key press
        key = getkey()

        if key == keys.RIGHT:
            return 3

        if key == keys.UP:
            return 1

        if key == keys.LEFT:
            return 4

        if key == keys.DOWN:
            return 2

        return 0

    def analyze_action_output(self, observation, reward, done, info):
        print("Reward: {}".format(reward))

    @staticmethod
    def position(agent_info):
        x, y, z = (*agent_info['brain_info'].vector_observations[0],)
        return x, y, z


class Stats:
    LAST_EXPLORATION_REWARDS = []

    def __init__(self):
        self.fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(2, 2, 1)
        self.ax2 = plt.subplot(2, 2, 2)
        self.ax3 = plt.subplot(2, 2, 3)

        ax1.title.set_text('Game view')

        self.im1 = ax1.imshow(numpy.zeros((84, 84, 3)))
        self.ax2.plot([])
        self.ax3.scatter([], [])

        # TODO - or plt.
        plt.ion()
        plt.show()

    def process_stats(self, agent, observation, reward, done, info, curiosity_reward):
        # Update exploration rewards
        self.LAST_EXPLORATION_REWARDS.append(curiosity_reward)
        self.LAST_EXPLORATION_REWARDS = self.LAST_EXPLORATION_REWARDS[-100:]

        # Update visualisation
        self.im1.set_data(observation[:, :, :])

        self.ax2.clear()
        self.ax2.plot(self.LAST_EXPLORATION_REWARDS)

        # self.ax3.clear()
        x, _, z = agent.position(info)
        self.ax3.scatter(x, z)

        self.ax2.title.set_text('Exploration reward')
        # self.ax3.title.set_text('Exploration map')

        self.fig.canvas.draw_idle()
        plt.pause(0.05)


class Trainer:
    def __init__(self, env, enable_stats=True):
        self.env = env
        self.agent = Agent(self.env)
        self.stats = None
        if enable_stats:
            self.stats = Stats()

    def train(self, episodes=10, episode_steps=50000):
        for episode in range(episodes):
            observation = env.reset()

            for step in range(episode_steps):
                action = self.agent.get_action(observation)

                observation, reward, done, info = env.step(action)

                self.agent.analyze_action_output(observation, reward, done, info)

                if self.stats:
                    self.stats.process_stats(self.agent, observation, reward, done, info, reward)

                if reward < 0.002 or done:
                    break


if __name__ == '__main__':
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    env = UnityEnv(
        config["env_filename"],
        config["env_worker_id"],
        use_visual=True,
        uint8_visual=True,
        no_graphics=False,
    )
    try:
        trainer = Trainer(env, enable_stats=config["enable_stats"])
        trainer.train(episodes=config["episodes"], episode_steps=config["episode_steps"])
    except Exception:
        env.close()
        raise
