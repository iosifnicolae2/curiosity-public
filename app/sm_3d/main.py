#!/usr/bin/env python3
from collections import namedtuple, defaultdict

import torch
import numpy
import yaml

import matplotlib.pyplot as plt

from getkey import getkey, keys
from gym_unity.envs import UnityEnv

from app.sm_3d.models import Memory, PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ManualControlAgent:
    def __init__(self, env):
        self.env = env

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
        if observation[0].shape == (256, 256, 3):
            self.im1.set_data(observation[0][:, :, :])

        if observation[0].shape == (256, 256, 1):
            self.im1.set_data(observation[0][:, :, 0])

        self.ax2.clear()
        self.ax2.plot(self.LAST_EXPLORATION_REWARDS)

        # self.ax3.clear()
        x, _, z = (*info['batched_step_result'].obs[3][0],)
        self.ax3.scatter(x, z)

        self.ax2.title.set_text('Exploration reward')
        # self.ax3.title.set_text('Exploration map')

        if len(observation) > 0 and observation[0].shape == (84, 84, 3):
            self.im1.set_data(observation[0][:, :, :])

        if len(observation) > 1 and observation[1].shape == (84, 84, 3):
            self.im2.set_data(observation[1][:, :, :])

        self.fig.canvas.draw_idle()
        plt.pause(0.05)


class Trainer:
    def __init__(
            self,
            env,
            config,
    ):
        self.env = env
        self.stats = None
        if config.enable_stats:
            self.stats = Stats()

        # Model hyperparams
        self.config = config

        #############################################

        if config.random_seed:
            torch.manual_seed(config.random_seed)
            env.seed(config.random_seed)

        self.memory = Memory(config)
        self.ppo = PPO(config)

    def train(self):
        # logging variables
        episode_reward = 0
        running_reward = 0
        avg_length = 0
        timestep = 0

        # training loop
        for i_episode in range(1, self.config.max_episodes + 1):
            state = env.reset()
            self.memory.clear_memory()

            reward, done, info, action, action_log_prob = 0, False, None, None, None
            t = 0

            for t in range(self.config.max_timesteps):
                timestep += 1

                # Save previous experience
                self.memory.save_signals(state, reward, done, info, action, action_log_prob)

                # Running policy_old:
                action, action_log_prob = self.ppo.policy_old.act(self.memory)

                state, reward, done, info = env.step(action)

                if self.stats:
                    self.stats.process_stats(state, reward, done, info)

                # update if its time
                if timestep % self.config.update_timestep == 0:
                    self.ppo.update(self.memory)
                    timestep = 0

                running_reward += reward
                episode_reward += reward

                if t % self.config.log_interval_timestamps == 0:
                    print(running_reward)
                    running_reward = 0

                if self.config.render:
                    env.render()
                if done:
                    break

            avg_length += t
            # stop training if avg_reward > solved_reward
            if episode_reward > self.config.solved_reward:
                print("########## Solved! ##########")
                torch.save(self.ppo.policy_old.state_dict(), './model_final.pth')
                break

            # logging
            if i_episode % self.config.log_interval == 0:
                avg_length = int(avg_length / self.config.log_interval)

                print('Episode {} \t avg length: {} \t episode_reward: {}'.format(i_episode, avg_length, episode_reward))

                torch.save(self.ppo.policy_old.state_dict(), 'app/sm_3d/saved_models/model_episode_{}.pth'.format(i_episode))

                episode_reward = 0
                running_reward = 0
                avg_length = 0

    def manual_control(self, episodes=10, episode_steps=50000):
        agent = ManualControlAgent(self.env)
        for episode in range(episodes):
            observation = env.reset()

            for step in range(episode_steps):
                action = agent.get_action(observation)

                observation, reward, done, info = env.step([action])

                agent.analyze_action_output(observation, reward, done, info)

                if self.stats:
                    self.stats.process_stats(observation, reward, done, info)

                if reward < 0.002 or done:
                    break

    def load_model(self, path):
        self.ppo.policy_old.load_state_dict(torch.load(path))

    def evaluate(self):
        # logging variables
        total_reward = 0
        running_reward = 0
        avg_length = 0
        timestep = 0

        # training loop
        for i_episode in range(1, self.config.max_episodes + 1):
            state = env.reset()
            reward, done, info, action, action_log_prob = 0, False, None, None, None
            t = 0
            total_reward = 0

            for t in range(self.config.max_timesteps):
                timestep += 1

                # Save previous experience
                self.memory.save_signals(state, reward, done, info, action, action_log_prob)

                # Running policy_old:
                action, action_log_prob = self.ppo.policy_old.act(self.memory)

                state, reward, done, info = env.step(action)

                if self.stats:
                    self.stats.process_stats(state, reward, done, info)

                running_reward += reward

                if t % self.config.log_interval_timestamps == 0:
                    print(running_reward)
                    running_reward = 0

                if self.config.render:
                    env.render()
                if done:
                    break

            avg_length += t

            # logging
            if i_episode % self.config.log_interval == 0:
                avg_length = int(avg_length / self.config.log_interval)
                running_reward = int((running_reward / self.config.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

                total_reward = 0
                running_reward = 0
                avg_length = 0


if __name__ == '__main__':
    with open('app/sm_3d/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config = namedtuple(
        'Struct',
        config.keys(),
    )(*config.values())

    env = UnityEnv(
        config.env_filename,
        config.env_worker_id,
        use_visual=True,
        uint8_visual=True,
        allow_multiple_visual_obs=True,
    )
    print('ok')
    try:
        trainer = Trainer(
            env,
            config
        )
        operations = config.operations

        if len(config.model_path) > 0 and 'clean' not in operations:
            trainer.load_model(config.model_path)

        if 'manual' in operations:
            trainer.manual_control()

        if 'train' in operations:
            trainer.train()

        if 'evaluate' in operations:
            trainer.evaluate()

    except Exception:
        env.close()
        raise

    env.close()
