import gym
import numpy as np
from marlgrid.envs import *

env = gym.make('MarlGrid-3AgentEmpty9x9-v0')

"""
>>> env.action_space
Tuple(Discrete(7), Discrete(7), Discrete(7))
>>> env.observation_space
Tuple(Box(56, 56, 3), Box(56, 56, 3), Box(56, 56, 3))
"""

N = 1  # No of episodes
n = len(env.agents) # No of agents
obs = env.reset()
for i in range(N):
    total_rewards = np.zeros(n)
    steps = 0
    while True:
        action = env.action_space.sample()
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
        steps += 1
        env.render()
        if done:
            obs = env.reset()
            break
    print(f'Episode: {i+1} | Total rewards: {total_rewards} | Steps: {steps}')