#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt
import numpy

from getkey import getkey, keys
from gym_unity.envs import UnityEnv

env = UnityEnv("unity-game.app", 1, use_visual=True, uint8_visual=True, no_graphics=False)

MEMORY = []


def save_curiosity_vect_obs(agent_observation):
    MEMORY.append(agent_observation)


def calculate_curiosity_reward(agent_observation):
    known_position = sum([1 for i in MEMORY if i.get_position_str() == agent_observation.get_position_str()])
    reward = 0

    if known_position > 0:
        reward += 0.1/known_position
    else:
        reward += 1

    return reward


def get_action(previous_observation, available_actions):

    # return available_actions.sample()

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


def get_agent_x_positions():
    for agent_observation in MEMORY:
        yield agent_observation.x


def get_agent_y_positions():
    for agent_observation in MEMORY:
        yield agent_observation.y


def get_agent_z_positions():
    for agent_observation in MEMORY:
        yield agent_observation.z


def train(episodes=10, episode_steps=50000):
    available_actions = env.action_space

    CURIOSITY_REWARDS = []

    fig = plt.figure(figsize=(8, 8))
    im1 = plt.subplot(2, 2, 1)
    im2 = plt.subplot(2, 2, 2)
    im3 = plt.subplot(2, 2, 3)

    im1.title.set_text('Game view')

    image_plot_1 = im1.imshow(numpy.zeros((84, 84, 3)))
    plt.ion()
    plt.show()

    for episode in range(episodes):
        observation = env.reset()
        episode_rewards = 0
        previous_observation = observation

        for step in range(episode_steps):
            action = get_action(previous_observation, available_actions)
            observation, reward, done, info = env.step(action)

            agent_observation = AgentObservation(info)

            curiosity_reward = calculate_curiosity_reward(agent_observation)
            CURIOSITY_REWARDS.append(curiosity_reward)
            CURIOSITY_REWARDS = CURIOSITY_REWARDS[-100:]

            episode_rewards += curiosity_reward
            print("curiosity_reward: {}".format(curiosity_reward))

            save_curiosity_vect_obs(agent_observation)
            previous_observation = observation

            # Update visualisation
            image_plot_1.set_data(previous_observation[:, :, :])
            im2.clear()
            im2.plot(CURIOSITY_REWARDS)
            im3.clear()
            im3.scatter(list(get_agent_x_positions()), list(get_agent_z_positions()))

            im2.title.set_text('Curiosity reward')
            im3.title.set_text('Exploration map')

            plt.pause(0.01)

            if curiosity_reward < 0.002:
                done = True

            if done:
                break

        print("Total reward this episode: {}".format(episode_rewards))

    env.close()


if __name__ == '__main__':
    train()
