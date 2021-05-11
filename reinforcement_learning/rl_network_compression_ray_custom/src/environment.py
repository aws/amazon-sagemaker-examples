from __future__ import absolute_import, division, print_function

from tempfile import mkstemp

import gym
import numpy as np
from gym.spaces import Box, MultiBinary
from tensorflow_resnet import NetworkCompression


class Compression(gym.Env):
    """Example of a network compression env which simulates layer removal and shrinkage.
    The reward is based on compression ratio between reference and current network, as well
    as the accuracy metric.
    It implements reset and step methods needed to interface with the gym envirnoment
    """

    def __init__(self, config={}):
        """Init function. This first defines the network compression model that implements layer
        removal/shrinkage and training to get reward. It also obtains the
        number of layers, and the network state shape,
        number_of_layers x number_of_parameters_for_each_layer. This method also defines
        action_space and observation_space for Gym environment based on these inputs
        """

        _, fileprefix = mkstemp()
        idx = fileprefix.rfind("/")
        fileprefix = fileprefix[idx + 1 :]
        self.network = NetworkCompression(prefix=fileprefix)
        self.num_actions = self.network.get_action_space()
        self.input_shape = self.network.get_observation_space()
        self.reward = self.network.get_reward()
        self.done = False

        self.action_space = Box(low=0, high=1, shape=(self.num_actions,), dtype=np.uint8)
        self.observation_space = Box(
            low=0, high=10000, shape=(self.input_shape[0] * self.input_shape[1],), dtype=np.uint8
        )

        self.cur_pos = self._get_current_pos_in_1d()

        return

    def _get_current_pos_in_1d(self):
        """This is an helper function that obtains the state of the network in
        number_of_layers x number_of_parameters_for_each_layer. However, coach does not
        like 2D array as input and hence recast it to 1D array"""
        pos = self.network.get_current_pos()
        pos = np.squeeze(np.reshape(pos, (self.input_shape[0] * self.input_shape[1], 1)))

        return pos

    def reset(self):
        """Gym reset interface. Sets the current position of the simulation
        which is always the parent network state"""
        self.cur_pos = self._get_current_pos_in_1d()

        return self.cur_pos

    def step(self, actions):
        """Gym step interface. Receives a set of binary actions (1d array)
        that determines whether to keep/remove each layer in the child network.
        This method also determines the reward obtained based on the action. Since
        we run only one step, the calculated reward is the final reward and we are done
        with an episode

        Args: 1D set of binary values
        """
        assert len(actions) == self.num_actions
        actions = np.around(actions)
        actions = np.clip(actions, 0, 1)
        self.done = self.network.perform_actions(actions)
        self.cur_pos = self._get_current_pos_in_1d()
        self.reward = self.network.get_reward()

        return self.cur_pos, self.reward, self.done, {}
