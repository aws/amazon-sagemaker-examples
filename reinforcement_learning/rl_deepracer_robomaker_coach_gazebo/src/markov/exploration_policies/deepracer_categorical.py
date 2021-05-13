from typing import List

import numpy as np
from rl_coach.core_types import ActionType, RunPhase
from rl_coach.exploration_policies.exploration_policy import (
    DiscreteActionExplorationPolicy,
    ExplorationParameters,
)
from rl_coach.spaces import ActionSpace


class DeepRacerCategoricalParameters(ExplorationParameters):
    def __init__(self, use_stochastic_evaluation_policy=False):
        super().__init__()
        self.use_stochastic_evaluation_policy = use_stochastic_evaluation_policy

    @property
    def path(self):
        return "markov.exploration_policies.deepracer_categorical:DeepRacerCategorical"


class DeepRacerCategorical(DiscreteActionExplorationPolicy):
    """
    Categorical exploration policy is intended for discrete action spaces. It expects the action values to
    represent a probability distribution over the action, from which a single action will be sampled.
    In evaluation, the action that has the highest probability will be selected. This is particularly useful for
    actor-critic schemes, where the actors output is a probability distribution over the actions.
    """

    def __init__(self, action_space: ActionSpace, use_stochastic_evaluation_policy: bool = False):
        """
        :param action_space: the action space used by the environment
        """
        super().__init__(action_space)
        self.use_stochastic_evaluation_policy = use_stochastic_evaluation_policy

    def get_action(self, action_values: List[ActionType]) -> (ActionType, List[float]):
        if self.phase == RunPhase.TRAIN or self.use_stochastic_evaluation_policy:
            # choose actions according to the probabilities
            action = np.random.choice(self.action_space.actions, p=action_values)
            return action, action_values
        else:
            # take the action with the highest probability
            action = np.argmax(action_values)
            one_hot_action_probabilities = np.zeros(len(self.action_space.actions))
            one_hot_action_probabilities[action] = 1

            return action, one_hot_action_probabilities

    def get_control_param(self):
        return 0
