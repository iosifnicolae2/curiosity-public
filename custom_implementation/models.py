#!/usr/bin/env python3
import itertools

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.distributions import Categorical

import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.vector_states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.vector_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorModel(nn.Module):
    MEMORY_SAMPLES = 50
    CNN_OUTPUT_CLASSES = 60
    VECTOR_STATE_LAYERS_OUTPUTS = 60
    LAST_VECTOR_STATES_LAYERS_OUTPUTS = 60

    def __init__(self, state_dim, vector_state_dim, action_dim, n_latent_var):
        super().__init__()
        self.conv_layers = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=self.CNN_OUTPUT_CLASSES,
        )

        self.vector_state_layers = nn.Sequential(
            nn.Linear(vector_state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, self.VECTOR_STATE_LAYERS_OUTPUTS),
        )

        self.past_vector_states_layers = nn.Sequential(
            nn.Linear(vector_state_dim * self.MEMORY_SAMPLES, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, self.LAST_VECTOR_STATES_LAYERS_OUTPUTS),
        )

        value_inputs = self.CNN_OUTPUT_CLASSES + self.VECTOR_STATE_LAYERS_OUTPUTS + self.LAST_VECTOR_STATES_LAYERS_OUTPUTS
        self.action_layer = nn.Sequential(
            nn.Linear(value_inputs, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, vector_state, past_vector_states):
        # IMPORTANT!
        # len(past_vector_states) should be equal to self.MEMORY_SAMPLES

        conv_layers_output = self.conv_layers(input)[0]
        vector_state_output = self.vector_state_layers(vector_state)
        past_vector_state_output = self.past_vector_states_layers(past_vector_states)

        x = torch.cat((conv_layers_output, vector_state_output, past_vector_state_output), -1)

        return self.action_layer(x)


class CriticModel(nn.Module):
    MEMORY_SAMPLES = 50
    CNN_OUTPUT_CLASSES = 60
    VECTOR_STATE_LAYERS_OUTPUTS = 60
    LAST_VECTOR_STATES_LAYERS_OUTPUTS = 60

    def __init__(self, state_dim, vector_state_dim, action_dim, n_latent_var):
        super().__init__()
        self.conv_layers = models.DenseNet(
            growth_rate=32,
            block_config=(3, 6, 4),
            num_init_features=64,
            num_classes=self.CNN_OUTPUT_CLASSES,
        )

        self.vector_state_layers = nn.Sequential(
            nn.Linear(vector_state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, self.VECTOR_STATE_LAYERS_OUTPUTS),
        )

        self.past_vector_states_layers = nn.Sequential(
            nn.Linear(vector_state_dim * self.MEMORY_SAMPLES, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, self.LAST_VECTOR_STATES_LAYERS_OUTPUTS),
        )

        value_inputs = self.CNN_OUTPUT_CLASSES + self.VECTOR_STATE_LAYERS_OUTPUTS + self.LAST_VECTOR_STATES_LAYERS_OUTPUTS
        self.value_layer = nn.Sequential(
            nn.Linear(value_inputs, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, input, vector_state, past_vector_states):
        # IMPORTANT!
        # len(past_vector_states) should be equal to self.MEMORY_SAMPLES

        conv_layers_output = self.conv_layers(input)[0]
        vector_state_output = self.vector_state_layers(vector_state)
        past_vector_state_output = self.past_vector_states_layers(past_vector_states)

        x = torch.cat((conv_layers_output, vector_state_output, past_vector_state_output), -1)

        return self.value_layer(x)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, vector_state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = ActorModel(state_dim, vector_state_dim, action_dim, n_latent_var)
        print("self.action_layer: ", self.action_layer)

        # critic
        self.value_layer = CriticModel(state_dim, vector_state_dim, action_dim, n_latent_var)
        print("self.value_layer: ", self.value_layer)

    def forward(self):
        raise NotImplementedError

    def act(self, state, vector_state, memory):
        # Use only one image feed
        state = state[0]

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        state = Image.fromarray(state)
        state = preprocess(state)

        # create a mini-batch as expected by the model
        input_batch = state.unsqueeze(0)
        input_batch = input_batch.to(device)

        require_more_x_elements = ActorModel.MEMORY_SAMPLES - len(memory.vector_states)
        preprocessed_vector_states = memory.vector_states
        if require_more_x_elements > 0:
            empty_array = torch.zeros((require_more_x_elements, 3))
            preprocessed_vector_states.extend(empty_array)
        preprocessed_vector_states = preprocessed_vector_states[:ActorModel.MEMORY_SAMPLES]
        # flat the list
        preprocessed_vector_states = list(itertools.chain.from_iterable(preprocessed_vector_states))
        vector_states_tensor = torch.tensor(preprocessed_vector_states).float().to(device)

        vector_state = torch.tensor(vector_state).float().to(device)

        action_probs = self.action_layer(input_batch, vector_state, vector_states_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.vector_states.append(vector_state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, vector_state, memory, action):
        action_probs = self.action_layer(state, vector_state, memory)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state, vector_state, memory)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, vector_state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, vector_state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, vector_state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_vector_states = torch.stack(memory.vector_states).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_vector_states[-1], old_vector_states[:-1], old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
