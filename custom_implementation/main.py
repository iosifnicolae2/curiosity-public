#!/usr/bin/env python3
import torch
import numpy
import yaml

import matplotlib.pyplot as plt

from getkey import getkey, keys
from gym_unity.envs import UnityEnv

from custom_implementation.models import Memory, PPO


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
        # print("Reward: {}".format(reward))
        pass

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
        ax4 = plt.subplot(2, 2, 4)

        ax1.title.set_text('Game view')

        self.im1 = ax1.imshow(numpy.zeros((256, 256, 3)), cmap='gray', vmin=0, vmax=255)
        self.ax2.plot([])
        self.ax3.scatter([], [])
        self.im2 = ax4.imshow(numpy.zeros((256, 256, 3)), cmap='gray', vmin=0, vmax=255)

        # TODO - or plt.
        plt.ion()
        plt.show()

    def process_stats(self, observation, reward, done, info, curiosity_reward):
        # Update exploration rewards
        self.LAST_EXPLORATION_REWARDS.append(curiosity_reward)
        self.LAST_EXPLORATION_REWARDS = self.LAST_EXPLORATION_REWARDS[-100:]

        # Update visualisation
        if observation[0].shape == (256, 256, 3):
            self.im1.set_data(observation[0][:, :, :])

        if observation[0].shape == (256, 256, 1):
            self.im1.set_data(observation[0][:, :, 0])

        self.ax2.clear()
        self.ax2.plot(self.LAST_EXPLORATION_REWARDS)

        # self.ax3.clear()
        x, _, z = (*info['brain_info'].vector_observations[0],)
        self.ax3.scatter(x, z)

        self.ax2.title.set_text('Exploration reward')
        # self.ax3.title.set_text('Exploration map')

        if len(observation) > 1 and observation[1].shape == (256, 256, 3):
            self.im2.set_data(observation[1][:, :, :])

        if len(observation) > 1 and observation[1].shape == (256, 256, 1):
            self.im2.set_data(observation[1][:, :, 0])

        self.fig.canvas.draw_idle()
        plt.pause(0.05)


class Trainer:
    def __init__(
            self,
            env,
            action_dim=4,
            render=False,
            solved_reward=230,  # stop training if avg_reward > solved_reward
            log_interval_timestamps=20,  # print avg reward in the interval
            log_interval=20,  # print avg reward in the interval
            max_episodes=50000,  # max training episodes
            max_timesteps=300,  # max timesteps in one episode
            n_latent_var=64,  # number of variables in hidden layer
            update_timestep=2000,  # update policy every n timesteps
            lr=0.002,
            betas=(0.9, 0.999),
            gamma=0.99,  # discount factor
            K_epochs=4,  # update policy for K epochs
            eps_clip=0.2,  # clip parameter for PPO
            random_seed=None,
            enable_stats=True,
    ):
        self.env = env
        self.stats = None
        if enable_stats:
            self.stats = Stats()

        # Model hyperparams
        self.action_dim = action_dim
        self.render = render
        self.solved_reward = solved_reward
        self.log_interval_timestamps = log_interval_timestamps
        self.log_interval = log_interval
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.n_latent_var = n_latent_var
        self.update_timestep = update_timestep
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.random_seed = random_seed

        self.state_dim = env.observation_space.shape
        self.vector_state_dim = 3  # TODO, we might need to find a better way to calculate this

        #############################################

        if random_seed:
            torch.manual_seed(random_seed)
            env.seed(random_seed)

        self.memory = Memory()
        self.ppo = PPO(
            self.state_dim,
            self.vector_state_dim,
            self.action_dim,
            self.n_latent_var,
            self.lr,
            self.betas,
            self.gamma,
            self.K_epochs,
            self.eps_clip,
        )

    def train(self):
        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            state = env.reset()
            vector_state = [0, 0, 0]
            for t in range(self.max_timesteps):
                timestep += 1

                # Running policy_old:
                action = self.ppo.policy_old.act(state, vector_state, self.memory)
                state, reward, done, info = env.step(action)
                vector_state = info['brain_info'].vector_observations[0]
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                if self.stats:
                    self.stats.process_stats(state, reward, done, info, reward)

                # update if its time
                if timestep % self.update_timestep == 0:
                    self.ppo.update(self.memory)
                    self.memory.clear_memory()
                    timestep = 0

                running_reward += reward

                if t % self.log_interval_timestamps == 0:
                    print(running_reward)

                if self.render:
                    env.render()
                if done:
                    break

            avg_length += t
            print(running_reward)
            # stop training if avg_reward > solved_reward
            if running_reward > (self.log_interval * self.solved_reward):
                print("########## Solved! ##########")
                torch.save(self.ppo.policy.state_dict(), './model_final.pth')
                break

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

                torch.save(self.ppo.policy.state_dict(), './model_episode_{}.pth'.format(i_episode))

                running_reward = 0
                avg_length = 0

    def manual_control(self, episodes=10, episode_steps=50000):

        agent = ManualControlAgent(self.env)
        for episode in range(episodes):
            observation = env.reset()

            for step in range(episode_steps):
                action = agent.get_action(observation)

                observation, reward, done, info = env.step(action)

                agent.analyze_action_output(observation, reward, done, info)

                if self.stats:
                    self.stats.process_stats(observation, reward, done, info, reward)

                if reward < 0.002 or done:
                    break

    def load_model(self, path):
        self.ppo.policy.load_state_dict(torch.load(path))

    def evaluate(self):
        # logging variables
        running_reward = 0
        avg_length = 0
        timestep = 0

        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            state = env.reset()
            for t in range(self.max_timesteps):
                timestep += 1

                # Running policy_old:
                action = self.ppo.policy_old.act(state, self.memory)
                state, reward, done, info = env.step(action)

                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                if self.stats:
                    self.stats.process_stats(state, reward, done, info, reward)

                running_reward += reward
                if self.render:
                    env.render()
                if done:
                    break

            avg_length += t
            print(running_reward)

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))

                torch.save(self.ppo.policy.state_dict(), './model_episode_{}.pth'.format(i_episode))

                running_reward = 0
                avg_length = 0


if __name__ == '__main__':
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    env = UnityEnv(
        config["env_filename"],
        config["env_worker_id"],
        use_visual=True,
        uint8_visual=True,
        allow_multiple_visual_obs=True,
    )
    print('ok')
    try:
        trainer = Trainer(
            env,
            action_dim=config["action_dim"],
            render=config["render"],
            solved_reward=config["solved_reward"],  # stop training if avg_reward > solved_reward
            log_interval_timestamps=config["log_interval_timestamps"],  # print avg reward in the interval
            log_interval=config["log_interval"],  # print avg reward in the interval
            max_episodes=config["max_episodes"],  # max training episodes
            max_timesteps=config["max_timesteps"],  # max timesteps in one episode
            n_latent_var=config["n_latent_var"],  # number of variables in hidden layer
            update_timestep=config["update_timestep"],  # update policy every n timesteps
            lr=config["lr"],
            betas=(config["betas_start"], config["betas_end"]),
            gamma=config["gamma"],  # discount factor
            K_epochs=config["K_epochs"],  # update policy for K epochs
            eps_clip=config["eps_clip"],  # clip parameter for PPO
            random_seed=config["random_seed"],
            enable_stats=config["enable_stats"],
        )
        operations = config["operations"]

        if 'manual' in operations:
            trainer.manual_control()

        if 'train' in operations:
            trainer.train()

        if 'evaluate' in operations:
            trainer.load_model(config["model_path"])
            trainer.evaluate()

    except Exception:
        env.close()
        raise

    env.close()
