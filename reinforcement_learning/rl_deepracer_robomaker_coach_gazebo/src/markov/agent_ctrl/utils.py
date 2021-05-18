"""This module should house utility methods for the agent control classes"""
import json
import logging
import math

import markov.agent_ctrl.constants as const
import numpy as np
from markov.agent_ctrl.constants import RewardParam
from markov.boto.s3.constants import ActionSpaceTypes, ModelMetadataKeys, TrainingAlgorithm
from markov.constants import SIMAPP_VERSION_1, SIMAPP_VERSION_2, SIMAPP_VERSION_3, SIMAPP_VERSION_4
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.metrics.constants import StepMetrics
from markov.multi_agent_coach.action_space_configs import (
    ClippedPPOActionSpaceConfig,
    SACActionSpaceConfig,
)
from markov.track_geom.constants import AgentPos, TrackNearDist, TrackNearPnts
from scipy.spatial.transform import Rotation

LOGGER = Logger(__name__, logging.INFO).get_logger()


def set_reward_and_metrics(
    reward_params,
    step_metrics,
    agent_name,
    pos_dict,
    track_data,
    data_dict,
    action,
    json_actions,
    car_pose,
):
    """Populates the reward_params and step_metrics dictionaries with the common
    metrics and parameters.
    reward_params - Dictionary containing the input parameters to the reward function
    step_metrics - Dictionary containing the metrics that are sent to s3
    agent_name - String of agent name
    pos_dict - Dictionary containing the agent position data, keys defined in AgentPos
    track_data - Object containing all the track information and geometry
    data_dict - Dictionary containing previous progress, steps, and start distance
    action - Integer containing the action to take
    json_actions - Dictionary that maps action into steering and angle
    car_pose - Gazebo Pose of the agent
    """
    try:
        # Check that the required keys are present in the dicts that are being
        # passed in, these methods will throw an exception if a key is missing
        RewardParam.validate_dict(reward_params)
        # model point and distance
        model_point = pos_dict[AgentPos.POINT.value]
        current_ndist = track_data.get_norm_dist(model_point)
        prev_index, next_index = track_data.find_prev_next_waypoints(current_ndist, normalized=True)
        # model progress starting at the initial waypoint
        reverse_dir = track_data.reverse_dir
        if reverse_dir:
            reward_params[const.RewardParam.LEFT_CENT.value[0]] = not reward_params[
                const.RewardParam.LEFT_CENT.value[0]
            ]
        current_progress = current_ndist - data_dict["start_ndist"]
        current_progress = compute_current_prog(current_progress, data_dict["prev_progress"])
        # Geat the nearest points
        nearest_pnts_dict = track_data.get_nearest_points(model_point)
        # Compute distance from center and road width
        nearest_dist_dict = track_data.get_nearest_dist(nearest_pnts_dict, model_point)
        # Compute the distance from the previous and next points
        distance_from_prev, distance_from_next = track_data.get_distance_from_next_and_prev(
            model_point, prev_index, next_index
        )
        # Compute which points are on the track
        wheel_on_track = track_data.points_on_track(pos_dict[AgentPos.LINK_POINTS.value])
        # Get the model orientation
        model_orientation = pos_dict[AgentPos.ORIENTATION.value]
        # Set the reward and metric parameters
        step_metrics[StepMetrics.STEPS.value] = reward_params[
            RewardParam.STEPS.value[0]
        ] = data_dict["steps"]
        reward_params[RewardParam.REVERSE.value[0]] = reverse_dir
        step_metrics[StepMetrics.PROG.value] = reward_params[
            RewardParam.PROG.value[0]
        ] = current_progress
        reward_params[RewardParam.CENTER_DIST.value[0]] = nearest_dist_dict[
            TrackNearDist.NEAR_DIST_CENT.value
        ]
        reward_params[RewardParam.PROJECTION_DISTANCE.value[0]] = (
            current_ndist * track_data.get_track_length()
        )
        reward_params[RewardParam.CLS_WAYPNY.value[0]] = [prev_index, next_index]
        reward_params[RewardParam.LEFT_CENT.value[0]] = (
            nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value]
            < nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]
        ) ^ (not track_data.is_ccw)
        reward_params[RewardParam.WAYPNTS.value[0]] = track_data.get_way_pnts()
        reward_params[RewardParam.TRACK_WIDTH.value[0]] = nearest_pnts_dict[
            TrackNearPnts.NEAR_PNT_IN.value
        ].distance(nearest_pnts_dict[TrackNearPnts.NEAR_PNT_OUT.value])
        reward_params[RewardParam.TRACK_LEN.value[0]] = track_data.get_track_length()
        step_metrics[StepMetrics.X.value] = reward_params[RewardParam.X.value[0]] = model_point.x
        step_metrics[StepMetrics.Y.value] = reward_params[RewardParam.Y.value[0]] = model_point.y
        step_metrics[StepMetrics.YAW.value] = reward_params[RewardParam.HEADING.value[0]] = (
            Rotation.from_quat(model_orientation).as_euler("zyx")[0] * 180.0 / math.pi
        )
        step_metrics[StepMetrics.CLS_WAYPNT.value] = (
            next_index if distance_from_next < distance_from_prev else prev_index
        )
        step_metrics[StepMetrics.TRACK_LEN.value] = track_data.get_track_length()
        step_metrics[StepMetrics.STEER.value] = reward_params[RewardParam.STEER.value[0]] = float(
            json_actions[ModelMetadataKeys.STEERING_ANGLE.value]
        )
        step_metrics[StepMetrics.THROTTLE.value] = reward_params[
            RewardParam.SPEED.value[0]
        ] = float(json_actions[ModelMetadataKeys.SPEED.value])
        step_metrics[StepMetrics.WHEELS_TRACK.value] = reward_params[
            RewardParam.WHEELS_ON_TRACK.value[0]
        ] = all(wheel_on_track)
        step_metrics[StepMetrics.ACTION.value] = action
        # set extra reward param for obstacle
        model_heading = reward_params[RewardParam.HEADING.value[0]]
        obstacle_reward_params = track_data.get_object_reward_params(
            agent_name, model_point, car_pose
        )
        if obstacle_reward_params:
            reward_params.update(obstacle_reward_params)
    except KeyError as ex:
        raise GenericRolloutException("Key {}, not found".format(ex))
    except Exception as ex:
        raise GenericRolloutException("Cannot compute reward and metrics: {}".format(ex))


def compute_current_prog(current_progress, prev_progress):
    """Returns the corrected current progress, this helper method checks to make user the
    current progress is sensible.
    current_progress - The current progress after taken a step
    prev_progress - The progress in the previous step
    """
    current_progress = 100 * current_progress
    # if agent moving in reversed direction
    if current_progress <= 0:
        current_progress += 100
    # cross finish line in normal direction
    if prev_progress > current_progress + 50.0:
        current_progress += 100.0
    # cross finish line in reversed direction
    if current_progress > prev_progress + 50.0:
        current_progress -= 100.0
    current_progress = min(current_progress, 100)
    return current_progress


def get_normalized_progress(current_progress, start_ndist):
    """
    Return normalized current progress with respect to START LINE of the track.

    Args:
        current_progress: current_progress to normalize (0 - 100)
        start_ndist: start_ndist to offset (0.0 - 1.0)

    Returns:
        normalized current progress with respect to START LINE of the track.

    """
    return (current_progress + start_ndist * 100) % 100


def send_action(velocity_pub_dict, steering_pub_dict, steering_angle, speed):
    """Publishes the given action to all the topics in the given dicts
    velocity_pub_dict - Dictionary containing all the velocity joints
    steering_pub_dict - Dictionary containing all the movable joints
    steering_angle - Desired amount, in radians, to move the movable joints by
    speed - Angular velocity which the velocity joints should rotate with
    """
    for _, pub in velocity_pub_dict.items():
        pub.publish(speed)

    for _, pub in steering_pub_dict.items():
        pub.publish(steering_angle)


def load_action_space(model_metadata):
    """Returns the action space object based on the training algorithm
       and action space type values passed in the model_metadata.json file

    Args:
        model_metadata (ModelMetadata): ModelMetadata object containing the details in the model metadata json file

    Returns:
        ActionSpace: RL Coach ActionSpace object corresponding to the type of action space
    """
    # get the json_actions
    json_actions = model_metadata.action_space
    if model_metadata.training_algorithm == TrainingAlgorithm.CLIPPED_PPO.value:
        action_space = ClippedPPOActionSpaceConfig(
            model_metadata.action_space_type
        ).get_action_space(json_actions)
    elif model_metadata.training_algorithm == TrainingAlgorithm.SAC.value:
        action_space = SACActionSpaceConfig(model_metadata.action_space_type).get_action_space(
            json_actions
        )
    else:
        log_and_exit(
            "Unknown training_algorithm value found while loading action space. \
            training_algorithm: {}".format(
                model_metadata.training_algorithm
            ),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    LOGGER.info("Action space from file: %s", json_actions)
    return action_space


def get_speed_factor(version):
    """Returns the velocity factor for a given physics version
    version (float): Sim app version for which to retrieve the velocity factor
    """
    if version >= SIMAPP_VERSION_3:
        return 2.77
    elif version == SIMAPP_VERSION_2:
        return 3.5
    elif version == SIMAPP_VERSION_1:
        return 1.0
    else:
        raise Exception("No velocity factor for given version")
