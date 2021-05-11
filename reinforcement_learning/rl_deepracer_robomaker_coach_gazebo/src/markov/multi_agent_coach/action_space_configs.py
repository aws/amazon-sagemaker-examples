"""This module contains the available action space configurations for the training algorithms"""
import abc
import logging

import numpy as np
from markov.boto.s3.constants import ActionSpaceTypes, ModelMetadataKeys, TrainingAlgorithm
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.multi_agent_coach.spaces import ScalableBoxActionSpace
from rl_coach.spaces import BoxActionSpace, DiscreteActionSpace


class ActionSpaceConfigInterface(object, metaclass=abc.ABCMeta):
    """This class defines an interface for action space configs, it defines
    the basic functionality required.
    """

    @abc.abstractmethod
    def get_action_space(self, json_actions):
        """Return the action space for the training algorithm

        Args:
            json_actions (dict): The json object containing the value of action_space key in model_metadata.json

        Returns:
            ActionSpace: Action space object for the particular training algorithm
        """
        raise NotImplementedError("Action space config must implement get_action_space method")


class ClippedPPOActionSpaceConfig(ActionSpaceConfigInterface):
    def __init__(self, action_space_type):
        """ClippedPPO training algorithm action space configuration

        Args:
            action_space_type (str): action space type used to identify what action space object to return
        """
        if action_space_type not in [
            ActionSpaceTypes.DISCRETE.value,
            ActionSpaceTypes.CONTINUOUS.value,
        ]:
            log_and_exit(
                "Unsupported action space type passed while defining ClippedPPOActionSpaceConfig. \
                action_space_type: {}".format(
                    action_space_type
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

        self.action_space_type = action_space_type

    def get_action_space(self, json_actions):
        """Return the action space for the training algorithm

        Args:
            json_actions (dict): The json object containing the value of action_space key in model_metadata.json

        Returns:
            ActionSpace: Action space object for the particular training algorithm
        """
        try:
            if self.action_space_type == ActionSpaceTypes.CONTINUOUS.value:
                # Setting the low and high values to -1 to +1 and scaling in the environment
                low = np.array([-1.0, -1.0])
                high = np.array([1.0, 1.0])
                # action_space for CLIPPED_PPO_CONTINUOUS training algorithm is assumed to be a dict with the format:
                # action_space:{steering_angle:{low:-30.0 , high:30.0}, speed:{low:0.1, high:4.0}}
                if (
                    json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                        ModelMetadataKeys.LOW.value
                    ]
                    >= json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                        ModelMetadataKeys.HIGH.value
                    ]
                    or json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.LOW.value]
                    >= json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.HIGH.value]
                ):
                    log_and_exit(
                        "Action space bounds are incorrect while defining ClippedPPOActionSpaceConfig. \
                        json_actions: {}".format(
                            json_actions
                        ),
                        SIMAPP_SIMULATION_WORKER_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_500,
                    )
                original_low = np.array(
                    [
                        json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                            ModelMetadataKeys.LOW.value
                        ],
                        json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.LOW.value],
                    ]
                )
                original_high = np.array(
                    [
                        json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                            ModelMetadataKeys.HIGH.value
                        ],
                        json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.HIGH.value],
                    ]
                )

                action_space = ScalableBoxActionSpace(
                    2,
                    low,
                    high,
                    default_action=0.5 * (high + low),
                    scale_action_space=True,
                    scaled_up_action_space_bounds={
                        ModelMetadataKeys.LOW.value: original_low,
                        ModelMetadataKeys.HIGH.value: original_high,
                    },
                )
            elif self.action_space_type == ActionSpaceTypes.DISCRETE.value:
                action_space = DiscreteActionSpace(
                    num_actions=len(json_actions),
                    default_action=next(
                        (
                            i
                            for i, v in enumerate(json_actions)
                            if v[ModelMetadataKeys.STEERING_ANGLE.value] == 0
                        ),
                        None,
                    ),
                )
            return action_space
        except Exception as ex:
            log_and_exit(
                "Error while getting action space in ClippedPPOActionSpaceConfig: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )


class SACActionSpaceConfig(ActionSpaceConfigInterface):
    def __init__(self, action_space_type):
        """SAC training algorithm action space configuration

        Args:
            action_space_type (str): action space type used to identify what action space object to return
        """
        if action_space_type not in [ActionSpaceTypes.CONTINUOUS.value]:
            log_and_exit(
                "Unsupported action space type passed while defining SACActionSpaceConfig. \
                action_space_type: {}".format(
                    action_space_type
                ),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

        self.action_space_type = action_space_type

    def get_action_space(self, json_actions):
        """Return the action space for the training algorithm

        Args:
            json_actions (dict): The json object containing the value of action_space key in model_metadata.json

        Returns:
            ActionSpace: Action space object for the particular training algorithm
        """
        try:
            # action_space for SAC training algorithm is assumed to be a dict with the format:
            # action_space:{steering_angle:{low:-30.0 , high:30.0}, speed:{low:0.1, high:4.0}}
            if (
                json_actions[ModelMetadataKeys.STEERING_ANGLE.value][ModelMetadataKeys.LOW.value]
                >= json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                    ModelMetadataKeys.HIGH.value
                ]
                or json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.LOW.value]
                >= json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.HIGH.value]
            ):
                log_and_exit(
                    "Action space bounds are incorrect while defining SACActionSpaceConfig. \
                    json_actions: {}".format(
                        json_actions
                    ),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_500,
                )
            low = np.array(
                [
                    json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                        ModelMetadataKeys.LOW.value
                    ],
                    json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.LOW.value],
                ]
            )
            high = np.array(
                [
                    json_actions[ModelMetadataKeys.STEERING_ANGLE.value][
                        ModelMetadataKeys.HIGH.value
                    ],
                    json_actions[ModelMetadataKeys.SPEED.value][ModelMetadataKeys.HIGH.value],
                ]
            )

            action_space = BoxActionSpace(2, low, high, default_action=0.5 * (high + low))
            return action_space
        except Exception as ex:
            log_and_exit(
                "Error while getting action space in SACActionSpaceConfig: {}".format(ex),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
