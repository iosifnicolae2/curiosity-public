#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy

from getkey import getkey, keys
from gym_unity.envs import UnityEnv


class Memory:
    __MEMORY = []

    def append_exploration_vect_obs(self, agent_observation):
        self.__MEMORY.append(agent_observation)

    def count_position_occurrences(self, agent_observation):
        return sum([
            1
            for i in self.__MEMORY
            if i.get_position_str() == agent_observation.get_position_str()
        ])

    def get_agent_x_positions(self):
        for agent_observation in self.__MEMORY:
            yield agent_observation.x

    def get_agent_y_positions(self):
        for agent_observation in self.__MEMORY:
            yield agent_observation.y

    def get_agent_z_positions(self):
        for agent_observation in self.__MEMORY:
            yield agent_observation.z


class AgentObservation:
    POSITION_ROUND_PRECISION = -1

    def __init__(self, agent_info):
        # agent position
        self.x, self.y, self.z = (*agent_info['brain_info'].vector_observations[0],)

    def get_position_str(self):
        return "{}:{}:{}".format(
            round(self.x, self.POSITION_ROUND_PRECISION),
            round(self.y, self.POSITION_ROUND_PRECISION),
            round(self.z, self.POSITION_ROUND_PRECISION),
        )


class Agent:
    def __init__(self, env):
        self.env = env
        self.memory = Memory()

    def get_reward(self, agent_observation):
        count_position_occurrences = self.memory.count_position_occurrences(agent_observation)
        reward = 0
        if count_position_occurrences > 0:
            reward += 0.1 / count_position_occurrences
        else:
            reward += 1

        return reward

    def get_action(self, current_observation):

        # return self.env.action_space.sample()

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

    def analyze_action(self, observation, reward, done, info):
        agent_observation = AgentObservation(info)

        curiosity_reward = self.get_reward(agent_observation)

        self.memory.append_exploration_vect_obs(agent_observation)

        return curiosity_reward


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

        self.ax3.clear()
        self.ax3.scatter(
            x=list(agent.memory.get_agent_x_positions()),
            y=list(agent.memory.get_agent_z_positions()),
        )

        self.ax2.title.set_text('Exploration reward')
        self.ax3.title.set_text('Exploration map')

        self.fig.canvas.draw_idle()
        plt.pause(0.05)


class Trainer:
    def __init__(self, env):
        self.env = env
        self.agent = Agent(self.env)
        self.stats = Stats()

    def train(self, episodes=10, episode_steps=50000):
        for episode in range(episodes):
            observation = env.reset()

            self.stats.process_stats(self.agent, observation, None, False, {}, None)

            for step in range(episode_steps):
                action = self.agent.get_action(observation)

                observation, reward, done, info = env.step(action)

                exploration_reward = self.agent.analyze_action(observation, reward, done, info)

                self.stats.process_stats(self.agent, observation, reward, done, info, exploration_reward)

                if exploration_reward < 0.002 or done:
                    break


if __name__ == '__main__':
    env = UnityEnv("unity-game.app", 1, use_visual=True, uint8_visual=True, no_graphics=False)
    try:
        trainer = Trainer(env)
        trainer.train(episode_steps=50)
    except Exception:
        env.close()
        raise
