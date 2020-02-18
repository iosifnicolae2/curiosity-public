#!/usr/bin/env python3
import threading
from collections import namedtuple, defaultdict
from datetime import datetime

import torch
import numpy
import yaml

import matplotlib.pyplot as plt

from getkey import getkey, keys

import gym
import gym_minigrid
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.wrappers import *

from app.sm_2d.models import Memory, PPO
from app.utils import ThreadWithReturnValue

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ManualControlAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, current_observation):

        # return self.env.action_space.sample()

        while True:
            # wait for a key press
            key = getkey()

            if key == keys.RIGHT:
                return MiniGridEnv.Actions.right

            if key == keys.UP:
                return MiniGridEnv.Actions.forward

            if key == keys.LEFT:
                return MiniGridEnv.Actions.left

            if key == 'p':
                return MiniGridEnv.Actions.pickup

            if key == 'd':
                return MiniGridEnv.Actions.drop

            if key == 't':
                return MiniGridEnv.Actions.toggle

            if key == 'f':
                return MiniGridEnv.Actions.done

    def analyze_action_output(self, observation, reward, done, info):
        print("Reward: {}".format(reward))
        pass

    @staticmethod
    def position(agent_info):
        x, y, z = (*agent_info['brain_info'].vector_observations[0],)
        return x, y, z


class Stats:
    LAST_EXPLORATION_REWARDS = []
    EPISODE_REWARDS = defaultdict(int)

    def __init__(self):
        self.episode = 0

        self.fig = plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(2, 3, 1)
        self.ax2 = plt.subplot(2, 3, 2)
        self.ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        self.ax5 = plt.subplot(2, 3, 5)

        ax1.title.set_text('Game view')

        self.im1 = ax1.imshow(numpy.zeros((256, 256, 3)), cmap='gray', vmin=0, vmax=255)
        self.ax2.plot([])
        self.ax3.scatter([], [])
        self.im2 = ax4.imshow(numpy.zeros((256, 256, 3)), cmap='gray', vmin=0, vmax=255)
        self.ax5.plot([])

        # TODO - or plt.
        plt.ion()
        plt.show()

    def process_stats(self, observation, reward, done, info):
        self.EPISODE_REWARDS[self.episode] += reward
        if done:
            self.episode += 1

            self.ax5.clear()
            self.ax5.title.set_text('Episodes reward')
            self.ax5.plot(list(self.EPISODE_REWARDS))

        # Update exploration rewards
        self.LAST_EXPLORATION_REWARDS.append(reward)
        self.LAST_EXPLORATION_REWARDS = self.LAST_EXPLORATION_REWARDS[-100:]

        # Update visualisation

        if observation.shape == (56, 56, 3):
            self.im1.set_data(observation[:, :, :])

        if observation[0].shape == (56, 56, 3):
            self.im1.set_data(observation[0][:, :, :])

        if observation[0].shape == (56, 56, 1):
            self.im1.set_data(observation[0][:, :, 0])

        self.ax2.clear()
        self.ax2.plot(self.LAST_EXPLORATION_REWARDS)

        # self.ax3.clear()
        if 'batched_step_result' in info:
            x, _, z = (*info['batched_step_result'].obs[3][0],)
            self.ax3.scatter(x, z)

        self.ax2.title.set_text('Exploration reward')
        # self.ax3.title.set_text('Exploration map')

        if len(observation) > 0 and observation[0].shape == (56, 56, 3):
            self.im1.set_data(observation[0][:, :, :])

        if len(observation) > 1 and observation[1].shape == (56, 56, 3):
            self.im2.set_data(observation[1][:, :, :])

        self.fig.canvas.draw_idle()
        plt.pause(0.05)


class Trainer:
    def __init__(
            self,
            config,
    ):
        self.stats = None
        if config.enable_stats:
            self.stats = Stats()

        # Model hyperparams
        self.config = config

        #############################################

        if config.random_seed:
            torch.manual_seed(config.random_seed)

        self.ppo = PPO(config)

    def train(self):
        # training loop
        remaining_episodes = self.config.max_episodes
        while remaining_episodes > 0:
            threads_num = self.config.threads
            threads = []

            for t in range(threads_num):
                threads.append(
                    ThreadWithReturnValue(target=self.collect_experiences, name='t{}'.format(t)),
                )

            for t in threads:
                t.start()

            for t in threads:
                memory = t.join()

                self.ppo.update(memory)

            self.save_policy()

            remaining_episodes -= threads_num
            print("remaining_episodes: {}".format(remaining_episodes))

    def collect_experiences(self):
        memory = Memory(self.config)
        memory.clear_memory()

        env = self.get_a_new_env()
        episode_reward = 0
        total_steps = 0
        start_date = datetime.now()
        env.reset()

        for t in range(self.config.memory_samples):
            # Running policy_old:
            action, action_log_prob = self.ppo.policy_old.act(memory)

            state, reward, done, info = env.step(action)

            if self.stats:
                self.stats.process_stats(state, reward, done, info)

            total_steps += 1
            episode_reward += reward

            # Save the experience
            memory.save_signals(state, reward, done, info, action, action_log_prob)

            if done:
                duration = (datetime.now() - start_date).total_seconds()
                frames_per_sec = total_steps/duration
                print('DONE. Episode_steps: {} \t Episode_reward: {} \t frames_per_sec: {}'.format(total_steps, episode_reward, frames_per_sec))
                return memory

            if self.config.render:
                env.render()

        env.close()
        duration = (datetime.now() - start_date).total_seconds()
        frames_per_sec = total_steps/duration

        print('Episode_steps: {} \t Episode_reward: {} \t frames_per_sec: {}'.format(total_steps, episode_reward, frames_per_sec))

        return memory

    def manual_control(self, episodes=10, episode_steps=50000):
        env = self.get_a_new_env()
        agent = ManualControlAgent(env)
        for episode in range(episodes):
            observation = env.reset()

            if self.config.render:
                env.render()

            for step in range(episode_steps):
                action = agent.get_action(observation)

                observation, reward, done, info = env.step(action)

                agent.analyze_action_output(observation, reward, done, info)

                if self.stats:
                    self.stats.process_stats(observation, reward, done, info)

                if self.config.render:
                    env.render()

                if done:
                    break

    def load_state_dict(self, state):
        self.ppo.policy_old.load_state_dict(state)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def train_single_core(self):
        # training loop
        remaining_episodes = self.config.max_episodes
        while remaining_episodes > 0:

            memory = self.collect_experiences()
            self.ppo.update(memory)
            self.save_policy()

            remaining_episodes -= 1
            print("remaining_episodes: {}".format(remaining_episodes))

    def evaluate(self):
        # evaluating loop
        remaining_episodes = self.config.max_episodes
        while remaining_episodes > 0:
            self.collect_experiences()
            remaining_episodes -= 1
            print("remaining_episodes: {}".format(remaining_episodes))

    def save_policy(self):
        torch.save(self.ppo.policy_old.state_dict(), './latest_model.pth')

    def get_a_new_env(self):
        env = gym.make(config.env)
        env = RGBImgPartialObsWrapper(env)  # Get pixel observations
        env = ImgObsWrapper(env)  # Get rid of the 'mission' field
        if config.random_seed:
            env.seed(config.random_seed)
        return env


if __name__ == '__main__':
    with open(r'app/sm_2d/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if 'train' in config['operations'] and config['threads'] > 1 and (config['render'] or config['enable_stats']):
        print("When training using multiple threads, the rendering will be disabled!")
        config['render'] = config['enable_stats'] = False

    config = namedtuple(
        'Struct',
        config.keys(),
    )(*config.values())

    try:
        trainer = Trainer(
            config
        )

        operations = config.operations
        if len(config.model_path) > 0 and 'clean' not in operations:
            trainer.load_model(config.model_path)

        if 'manual' in operations:
            trainer.manual_control()

        if 'train' in operations:
            if config.threads > 1:
                trainer.train()
            else:
                trainer.train_single_core()

        if 'evaluate' in operations:
            trainer.evaluate()

    except Exception:
        raise
