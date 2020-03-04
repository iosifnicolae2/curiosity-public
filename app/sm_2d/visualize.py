import torch

import gym
import gym_minigrid

from app.sm_2d.utils import get_model_dir, make_env
from app.sm_2d.agents import Agent

from app.sm_2d.env_registers import *

env = 'MiniGrid-SimpleCrossingS11N5-v0'
seed = 1
mem = False
text = False

shift = 0
model = 'PPO'
episodes = 1000000
argmax = False

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

env = make_env(env, seed)
for _ in range(shift):
    env.reset()
print("Environment loaded\n")

# Load agent


model_dir = get_model_dir(model)
agent = Agent(env.observation_space, env.action_space, model_dir, device, argmax)
print("Agent loaded\n")


# Create a window to view the environment
env.render('human')

for episode in range(episodes):
    obs = env.reset()

    while True:
        env.render('human')

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if done or env.window.closed:
            break

    if env.window.closed:
        break


if __name__ == '__main__':
    pass