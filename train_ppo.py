import gym
import numpy as np
from marlgrid.envs import *
from marlgrid.agents import GridAgentInterface
from marl.agents import RLAgents, PPOAgent



env = gym.make('MarlGrid-3AgentEmpty9x9-v0')

"""
>>> env.action_space
Tuple(Discrete(7), Discrete(7), Discrete(7))
>>> env.observation_space
Tuple(Box(56, 56, 3), Box(56, 56, 3), Box(56, 56, 3))
"""
# TODO: Add logging for episodes

num_episodes = 1000

# Interface between grid and agents
iface = GridAgentInterface()


# Create type of learner agents
learner_agents = [PPOAgent(iface.observation_space, iface.action_space) for _ in range(env.num_agents)]
agents = RLAgents(*learner_agents)

# Iterate over episodes
total_steps = 0
total_reward = 0
for ep_num in range(num_episodes):
    obs_array = env.reset()

    done = False
    ep_steps = 0
    ep_reward = 0

    while not done:
        # Get actions from agents
        action_array = agents.action_step(obs_array)

        next_obs_array, reward_array, done, _ = env.step(action_array)

        obs_array = next_obs_array

        # Update steps
        ep_steps += 1
        total_steps += 1

        # Update rewards
        ep_reward += np.sum(reward_array)
        total_reward += np.sum(reward_array)

