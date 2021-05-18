from typing import Dict, List, Union

import numpy as np
from markov.boto.s3.constants import ModelMetadataKeys
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from rl_coach.spaces import BoxActionSpace


class ScalableBoxActionSpace(BoxActionSpace):
    def __init__(
        self,
        shape: Union[int, np.ndarray],
        low: Union[None, int, float, np.ndarray] = -np.inf,
        high: Union[None, int, float, np.ndarray] = np.inf,
        descriptions: Union[None, List, Dict] = None,
        default_action: np.ndarray = None,
        scale_action_space: bool = False,
        scaled_up_action_space_bounds: Dict = {
            ModelMetadataKeys.LOW.value: -np.inf,
            ModelMetadataKeys.HIGH.value: np.inf,
        },
    ):
        """This class extends BoxActionSpace and adds ability to scale the actions

        Args:
            shape (Union[int, np.ndarray]): int or array value of the shape of the action space
            low (Union[None, int, float, np.ndarray], optional): higher bound of the action space. Defaults to -np.inf.
            high (Union[None, int, float, np.ndarray], optional): higher bound of the action space. Defaults to np.inf.
            descriptions (Union[None, List, Dict], optional): description set for each action value. Defaults to None.
            default_action (np.ndarray, optional): default action value. Defaults to None.
            scale_action_space (bool, optional): boolean value to indicate if scaling needs to be done. Defaults to False.
            scaled_up_action_space_bounds (Dict, optional): dictionary defining the scaled up minimum and maximum bounds.
                                                            Defaults to {ModelMetadataKeys.LOW.value: -np.inf, ModelMetadataKeys.HIGH.value: np.inf}.
        """
        super().__init__(shape, low, high, descriptions, default_action)
        self.scale_action_space = scale_action_space
        self.scaled_up_action_space_bounds = scaled_up_action_space_bounds

    def scale_action_values(self, actions):
        """Return the action space for the training algorithm

        Args:
            actions (list(float)): The list of actions that need to be scaled

        Returns:
            list(float): scaled/unscaled actions depending on the scale_action_space value set
        """
        if not self.scale_action_space:
            return actions
        scaled_actions = list()
        # Rescale each of the action in the actions list accoridng the bounds passed
        for action, low, high, scaled_low, scaled_high in zip(
            actions,
            self.low,
            self.high,
            self.scaled_up_action_space_bounds[ModelMetadataKeys.LOW.value],
            self.scaled_up_action_space_bounds[ModelMetadataKeys.HIGH.value],
        ):
            scaled_actions.append(self._scale_value(action, low, high, scaled_low, scaled_high))
        return scaled_actions

    def _scale_value(self, action, min_old, max_old, min_new, max_new):
        """Return the scaled action value from min_old,max_old to min_new,max_new

        Args:
            action (float): The action value to be scaled
            min_old (float): The minimum bound value before scaling
            max_old (float): The maximum bound value before scaling
            min_new (float): The minimum bound value after scaling
            max_new (float): The maximum bound value after scaling

        Returns:
            (float): scaled action value
        """
        if max_old == min_old:
            log_and_exit(
                "Unsupported minimum and maximum action space bounds for scaling values. \
                min_old: {}; max_old: {}".format(
                    min_old, max_old
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        return ((max_new - min_new) / (max_old - min_old)) * (action - min_old) + min_new
