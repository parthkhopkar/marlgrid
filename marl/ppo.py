# This module will have the class for the PPO learning agent and also the class for the ML model for the agent

import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.Layers import Input, Conv2D, Dense, Flatten

# TODO: Make sure that the observation space given to model is not a tuple

class PPOModel(tf.keras.Model):
    def __init__(
        self,
        observation_space,
        action_space,
        batch_size
        ):
        super(PPOModel, self).__init__()

        # TODO: Verify output formats of model
        
        input_shape = observation_space.shape
        self.conv1 = Conv2D(filters=8, kernel_size=3, strides=3, input_shape=input_shape, activation='relu')
        self.conv2 = Conv2D(filters=16, kernel_size=3, strides=1, input_shape=input_shape, activation='relu')
        self.conv3 = Conv2D(filters=16, kernel_size=3, strides=1, input_shape=input_shape, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(192, activation = 'tanh')
        self.dense2 = Dense(192, activation = 'tanh')
        self.mlp_pi = keras.Sequential([
            Dense(64, activation="relu", name="val_layer1"),
            Dense(64, activation="relu", name="val_layer2"),
            Dense(action_space.n, activation='linear')])
        self.mlp_val = keras.Sequential([
            Dense(64, activation="relu", name="val_layer1"),
            Dense(64, activation="relu", name="val_layer2"),
            Dense(1, activation='linear']))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        pi = self.mlp_pi(x)
        val = self.mlp_val(x)
        return pi, val


class PPOAgent():
    # Configuration and hyperparameters

    # Constructor
    def __init__(self, observation_space, action_space):
        # TODO: check these 
        num_inputs = observation_space
        num_ouputs = action_space

        ################################
        # Hyperparameters for Learning #
        ################################
        

        # Initialize the model
        self.ac = PPOModel(self.observation_space, self.action_space)
        self.training = True

        # Replay memory



    # Get actions based on observation
    def action_step(self, obs):
        # TODO: Check input dimensions
        pi, val = self.ac(obs)