# This module contains the RL agents which will be present in the environment
import numpy as np
import tensorflow as tf
from tensorflow import keras

class RLAgents():
    """Class for all RL agents present in the environment
    """
    def __init__(self, *agents):
        self.agents = list(agents)  # List of learner agents

    def get_actions(self, obs_array):
        return [agent.action_step(obs) for agent, obs in zip(self.agents, obs_array)]
        


