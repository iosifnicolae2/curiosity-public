#!/usr/bin/env python3
import itertools

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.distributions import Categorical

import torchvision.models as models
import torchvision.transforms as transforms

from custom_implementation.utils import push_to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, config):
        self.config = config

        self.initialize_data()

    def clear_memory(self):
        self.initialize_data()

    def initialize_data(self):
        self.actions = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.int)
        self.states = torch.zeros(1, self.config.memory_samples, 2, 3, 84, 84, dtype=torch.float)
        self.logprobs = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.float)
        self.vector_observations = torch.zeros(1, self.config.memory_samples, self.config.vector_observation_dim,
                                               dtype=torch.float)
        self.rewards = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.float)
        self.is_terminals = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.bool)

    def save_signals(self, state, reward, done, info, action, action_log_prob):
        push_to_tensor(self.rewards, torch.tensor(reward, dtype=torch.float)) if reward is not None else None
        push_to_tensor(self.is_terminals, torch.tensor(done, dtype=torch.bool)) if done is not None else None
        push_to_tensor(self.states, torch.tensor(self.preprocess_images(state), dtype=torch.float)) if state is not None else None
        push_to_tensor(self.logprobs, torch.tensor(action_log_prob, dtype=torch.float)) if action_log_prob is not None else None
        agent_position = info['batched_step_result'].obs[3][0] if info is not None else None
        push_to_tensor(self.vector_observations, torch.tensor(agent_position, dtype=torch.float)) if agent_position is not None else None
        push_to_tensor(self.actions, torch.tensor(action, dtype=torch.int)) if action is not None else None

    def to(self, device):
        self.actions = self.actions.to(device)
        self.states = self.states.to(device)
        self.logprobs = self.logprobs.to(device)
        self.vector_observations = self.vector_observations.to(device)
        self.rewards = self.rewards.to(device)
        self.is_terminals = self.is_terminals.to(device)
        return self

    @property
    def last_state_single_camera(self):
        return self.states[:, -1, 0]

    @property
    def latest_vector_observation(self):
        return self.vector_observations[:, -1]

    @property
    def previous_vector_observations(self):
        return self.vector_observations[:, -1 - self.config.memory_samples:-1]

    @property
    def last_action(self):
        return self.actions[:, -1]

    @staticmethod
    def preprocess_images(images):
        processed_images = None
        for img in images:
            processed_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(Image.fromarray(img)).unsqueeze(0)

            if processed_images is None:
                processed_images = processed_image
            else:
                processed_images = torch.cat((processed_images, processed_image), 0)

        return processed_images


class ActorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_layers = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=60,
        )

        self.old_vector_observations_layers = nn.Sequential(
            nn.Linear(self.config.vector_observation_dim * (self.config.memory_samples - 1), self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, 60),
        )

        self.current_vector_observations_layers = nn.Sequential(
            nn.Linear(self.config.vector_observation_dim, self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, 60),
        )

        self.action_layers = nn.Sequential(
            nn.Linear(180, self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, self.config.action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, memory):
        # Load data on GPU
        conv_layers_input = memory.last_state_single_camera.to(device)
        current_vector_observation = memory.latest_vector_observation.to(device)
        old_vector_observations = torch.flatten(memory.previous_vector_observations, start_dim=1).to(device)

        # Execute the model
        conv_layers_output = self.conv_layers(conv_layers_input)
        current_vector_observations_layers_output = self.current_vector_observations_layers(current_vector_observation)
        old_vector_observations_layers_output = self.old_vector_observations_layers(old_vector_observations)
        x = torch.cat((conv_layers_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)

        action_layers_output = self.action_layers(x)

        return action_layers_output


class CriticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_layers = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=60,
        )

        self.old_vector_observations_layers = nn.Sequential(
            nn.Linear(self.config.vector_observation_dim * (self.config.memory_samples - 1), self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, 60),
        )

        self.current_vector_observations_layers = nn.Sequential(
            nn.Linear(self.config.vector_observation_dim, self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, 60),
        )

        self.value_layers = nn.Sequential(
            nn.Linear(180, self.config.n_latent_var),
            nn.Tanh(),
            nn.Linear(self.config.n_latent_var, 1),
        )

    def forward(self, memory):
        # Load data on GPU
        conv_layers_input = memory.last_state_single_camera.to(device)
        current_vector_observation = memory.latest_vector_observation.to(device)
        old_vector_observations = torch.flatten(memory.previous_vector_observations, start_dim=1).to(device)

        # Execute the model
        conv_layers_output = self.conv_layers(conv_layers_input)[0]
        current_vector_observations_layers_output = self.current_vector_observations_layers(current_vector_observation)[0]
        old_vector_observations_layers_output = self.old_vector_observations_layers(old_vector_observations)[0]
        x = torch.cat((conv_layers_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)

        value_layers_output = self.value_layers(x)

        return value_layers_output


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()
        self.config = config
        self.action_layers = ActorModel(self.config)
        self.value_layers = CriticModel(self.config)

    def forward(self):
        raise NotImplementedError

    def act(self, memory):
        action_probs = self.action_layers(memory)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, memory):

        # Load data into GPU
        action = torch.tensor(memory.last_action).to(device)

        # Execute the model
        # input = [
        #     Input(self.config, state, info)
        #     for state, info in zip(memory.states[:-1], memory.info[:-1])
        # ]
        # TODO: implement batch processing

        # for state, info in zip(memory.states, memory.infos):
        #     input = Input(self.config, state, info)

        action_probs = self.action_layers(memory)
        dist = Categorical(action_probs)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layers(memory)

        # Preprocess the data
        state_value = torch.squeeze(state_value)

        return action_logprob, state_value, dist_entropy


class PPO:
    def __init__(self, config):
        self.config = config

        self.policy_old = ActorCritic(self.config).to(device)
        self.policy = ActorCritic(self.config).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr, betas=(self.config.betas_start, self.config.betas_end))
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards[0]), reversed(list(memory.is_terminals[0]))):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.config.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        gpu_memory = memory.to(device)

        # Optimize policy for K epochs:
        for _ in range(self.config.K_epochs):
            # Evaluating old actions and values
            # TODO: we can speed up this process by loading the memory in GPU one time, not for each K_epoch
            logprobs, state_values, dist_entropy = self.policy.evaluate(gpu_memory)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - gpu_memory.logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            # TODO
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
