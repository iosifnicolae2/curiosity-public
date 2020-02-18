#!/usr/bin/env python3
import torch
import torch.nn as nn

from PIL import Image
from torch.distributions import Categorical

import torchvision.models as models
import torchvision.transforms as transforms

from app.utils import push_to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, config):
        self.config = config

        self.initialize_data()

    def clear_memory(self):
        self.initialize_data()

    def initialize_data(self):
        self.actions = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.int).to(device)
        self.states = torch.zeros(1, self.config.memory_samples, 1, 3, self.config.image_width, self.config.image_height, dtype=torch.float).to(device)
        # self.states = torch.zeros(1, self.config.memory_samples, 2, 3, self.config.image_width, self.config.image_height, dtype=torch.float).to(device)
        self.logprobs = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.float).to(device)
        self.vector_observations = torch.zeros(1, self.config.memory_samples, self.config.vector_observation_dim,
                                               dtype=torch.float).to(device)
        self.rewards = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.float).to(device)
        self.is_terminals = torch.zeros(1, self.config.memory_samples, 1, dtype=torch.bool).to(device)

    def save_signals(self, state, reward, done, info, action, action_log_prob):
        if reward:
            self.rewards = push_to_tensor(self.rewards, torch.tensor(reward, dtype=torch.float).to(device))
        if done:
            self.is_terminals = push_to_tensor(self.is_terminals, torch.tensor(done, dtype=torch.bool).to(device))
        if state is not None:
            self.states = push_to_tensor(self.states, self.preprocess_images(state).float().to(device))
        if action_log_prob:
            self.logprobs = push_to_tensor(self.logprobs, action_log_prob.float().to(device))
        if info:
            agent_position = info['batched_step_result'].obs[3][0]
            self.vector_observations = push_to_tensor(self.vector_observations, torch.tensor(agent_position, dtype=torch.float).to(device))
        if action:
            self.actions = push_to_tensor(self.actions, torch.tensor(action, dtype=torch.int).to(device))

    # def to(self, device):
    #     self.actions = self.actions.to(device)
    #     self.states = self.states.to(device)
    #     self.logprobs = self.logprobs.to(device)
    #     self.vector_observations = self.vector_observations.to(device)
    #     self.rewards = self.rewards.to(device)
    #     self.is_terminals = self.is_terminals.to(device)
    #     return self

    @property
    def last_state_first_camera(self):
        return self.states[:, -1, 0]

    # @property
    # def last_state_second_camera(self):
    #     return self.states[:, -1, 1]

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
    def preprocess_images(output_img):
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(output_img)
        # torch.Size([1, 50, 1, 3, 84, 84])
        processed_images = img.unsqueeze_(0)
        return processed_images.unsqueeze_(0)


class ActorModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_layers1 = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=60,
        )

        # self.conv_layers2 = models.DenseNet(
        #     growth_rate=32,
        #     block_config=(3, 6, 4),
        #     num_init_features=64,
        #     num_classes=60,
        # )

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
        # self.action_layers = nn.Sequential(
        #     nn.Linear(240, self.config.n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(self.config.n_latent_var, self.config.action_dim),
        #     nn.Softmax(dim=-1)
        # )

    def forward(self, memory):
        # Load data on GPU
        conv_layers_input1 = memory.last_state_first_camera.to(device)
        # conv_layers_input2 = memory.last_state_second_camera.to(device)
        current_vector_observation = memory.latest_vector_observation.to(device)
        old_vector_observations = torch.flatten(memory.previous_vector_observations, start_dim=1).to(device)

        # Execute the model
        conv_layers1_output = self.conv_layers1(conv_layers_input1)
        # conv_layers2_output = self.conv_layers2(conv_layers_input2)
        current_vector_observations_layers_output = self.current_vector_observations_layers(current_vector_observation)
        old_vector_observations_layers_output = self.old_vector_observations_layers(old_vector_observations)
        x = torch.cat((conv_layers1_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)
        # x = torch.cat((conv_layers1_output, conv_layers2_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)

        action_layers_output = self.action_layers(x)

        return action_layers_output


class CriticModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_layers1 = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=60,
        )

        # self.conv_layers2 = models.DenseNet(
        #     growth_rate=32,
        #     block_config=(3, 6, 4),
        #     num_init_features=64,
        #     num_classes=60,
        # )

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

        # self.value_layers = nn.Sequential(
        #     nn.Linear(240, self.config.n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(self.config.n_latent_var, 1),
        # )

    def forward(self, memory):
        # Load data on GPU
        # Load data on GPU
        conv_layers_input1 = memory.last_state_first_camera.to(device)
        # conv_layers_input2 = memory.last_state_second_camera.to(device)
        current_vector_observation = memory.latest_vector_observation.to(device)
        old_vector_observations = torch.flatten(memory.previous_vector_observations, start_dim=1).to(device)

        # Execute the model
        conv_layers1_output = self.conv_layers1(conv_layers_input1)
        # conv_layers2_output = self.conv_layers2(conv_layers_input2)
        current_vector_observations_layers_output = self.current_vector_observations_layers(current_vector_observation)
        old_vector_observations_layers_output = self.old_vector_observations_layers(old_vector_observations)
        x = torch.cat((conv_layers1_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)
        # x = torch.cat((conv_layers1_output, conv_layers2_output, current_vector_observations_layers_output, old_vector_observations_layers_output), -1)

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
        action = memory.last_action.to(device)

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
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.config.K_epochs):
            # Evaluating old actions and values
            # TODO: we can speed up this process by loading the values from memory on GPU one time, not for each K_epoch
            logprobs, state_values, dist_entropy = self.policy.evaluate(memory)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - memory.logprobs.detach()).to(device)

            # Finding Surrogate Loss:
            advantages = (rewards - state_values.detach()).to(device)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages

            rewards_sum = torch.sum(rewards).to(device)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards_sum) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            # TODO
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
