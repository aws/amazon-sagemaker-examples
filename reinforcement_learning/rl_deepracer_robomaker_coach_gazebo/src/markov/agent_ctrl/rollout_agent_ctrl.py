'''This module implements concrete agent controllers for the rollout worker'''
import copy
import time
from collections import OrderedDict
import math
import numpy as np
import rospy
import logging

from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64
from shapely.geometry import Point
from markov.visualizations.reward_distributions import RewardDataPublisher

import markov.agent_ctrl.constants as const
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.agent_ctrl.utils import (set_reward_and_metrics,
                                     send_action, load_action_space, get_speed_factor,
                                     get_normalized_progress, Logger)
from markov.track_geom.constants import (AgentPos, TrackNearDist, ObstacleDimensions)
from markov.track_geom.track_data import FiniteDifference, TrackData
from markov.track_geom.utils import euler_to_quaternion, pose_distance, apply_orientation
from markov.metrics.constants import StepMetrics, EpisodeStatus
from markov.cameras.camera_manager import CameraManager
from markov.common import ObserverInterface
from markov.log_handler.deepracer_exceptions import RewardFunctionError, GenericRolloutException
from markov.reset.constants import AgentPhase, AgentCtrlStatus, AgentInfo
from markov.reset.utils import construct_reset_rules_manager
from markov.utils import get_racecar_idx
from markov.visual_effects.effects.blink_effect import BlinkEffect
from markov.constants import DEFAULT_PARK_POSITION
from markov.gazebo_tracker.trackers.get_link_state_tracker import GetLinkStateTracker
from markov.gazebo_tracker.trackers.get_model_state_tracker import GetModelStateTracker
from markov.gazebo_tracker.trackers.set_model_state_tracker import SetModelStateTracker
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.gazebo_tracker.constants import TrackerPriority

from rl_coach.core_types import RunPhase

logger = Logger(__name__, logging.INFO).get_logger()


class RolloutCtrl(AgentCtrlInterface, ObserverInterface, AbstractTracker):
    '''Concrete class for an agent that drives forward'''
    def __init__(self, config_dict, run_phase_sink, metrics):
        '''config_dict (dict): containing all the keys in ConfigParams
           run_phase_sink (RunPhaseSubject): Sink to receive notification of a change in run phase
           metrics (EvalMetrics/TrainingMetrics): Training or evaluation metrics
        '''
        # reset rules manager
        self._metrics = metrics
        self._is_continuous = config_dict[const.ConfigParams.IS_CONTINUOUS.value]
        self._reset_rules_manager = construct_reset_rules_manager(config_dict)
        self._ctrl_status = dict()
        self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.RUN.value
        self._config_dict = config_dict
        self._done_condition = config_dict.get(const.ConfigParams.DONE_CONDITION.value, any)
        self._number_of_resets = config_dict[const.ConfigParams.NUMBER_OF_RESETS.value]
        self._off_track_penalty = config_dict[const.ConfigParams.OFF_TRACK_PENALTY.value]
        self._collision_penalty = config_dict[const.ConfigParams.COLLISION_PENALTY.value]
        self._pause_duration = 0.0
        self._reset_count = 0
        self._curr_crashed_object_name = ''
        # simapp_version speed scale
        self._speed_scale_factor_ = get_speed_factor(config_dict[const.ConfigParams.VERSION.value])
        # Store the name of the agent used to set agents position on the track
        self._agent_name_ = config_dict[const.ConfigParams.AGENT_NAME.value]
        # Set start lane. This only support for two agents H2H race
        self._agent_idx_ = get_racecar_idx(self._agent_name_)
        # Get track data
        self._track_data_ = TrackData.get_instance()
        if self._agent_idx_ is not None:
            self._start_lane_ = self._track_data_.inner_lane \
                if self._agent_idx_ % 2 else self._track_data_.outer_lane
        else:
            self._start_lane_ = self._track_data_.center_line
        # Store the name of the links in the agent, this should be const
        self._agent_link_name_list_ = config_dict[const.ConfigParams.LINK_NAME_LIST.value]
        # Store the reward function
        self._reward_ = config_dict[const.ConfigParams.REWARD.value]
        # Create publishers for controlling the car
        self._velocity_pub_dict_ = OrderedDict()
        self._steering_pub_dict_ = OrderedDict()
        for topic in config_dict[const.ConfigParams.VELOCITY_LIST.value]:
            self._velocity_pub_dict_[topic] = rospy.Publisher(topic, Float64, queue_size=1)
        for topic in config_dict[const.ConfigParams.STEERING_LIST.value]:
            self._steering_pub_dict_[topic] = rospy.Publisher(topic, Float64, queue_size=1)
        #Create default reward parameters
        self._reward_params_ = const.RewardParam.make_default_param()
        #Create the default metrics dictionary
        self._step_metrics_ = StepMetrics.make_default_metric()
        # Dictionary of bools indicating starting position behavior
        self._start_pos_behavior_ = \
            {'change_start' : config_dict[const.ConfigParams.CHANGE_START.value],
             'alternate_dir' : config_dict[const.ConfigParams.ALT_DIR.value]}
        # Dictionary to track the previous way points
        self._prev_waypoints_ = {'prev_point' : Point(0, 0), 'prev_point_2' : Point(0, 0)}

        # Normalized distance of new start line from the original start line of the track.
        start_ndist = 0.0

        # Normalized start position offset w.r.t to start_ndist, which is the start line of the track.
        start_pos_offset = config_dict.get(const.ConfigParams.START_POSITION.value, 0.0)
        self._start_line_ndist_offset = start_pos_offset / self._track_data_.get_track_length()

        # Dictionary containing some of the data for the agent
        # - During the reset call, every value except start_ndist will get wiped out by self._clear_data
        #   (reset happens prior to every episodes begin)
        # - If self._start_line_ndist_offset is not 0 (usually some minus value),
        #   then initial current_progress suppose to be non-zero (usually some minus value) as progress
        #   suppose to be based on start_ndist.
        # - This will be correctly calculated by first call of utils.compute_current_prog function.
        #   As prev_progress will be initially 0.0 and physical position is not at start_ndist,
        #   utils.compute_current_prog will return negative progress if self._start_line_ndist_offset is negative value
        #   (meaning behind start line) and will return positive progress if self._start_line_ndist_offset is
        #   positive value (meaning ahead of start line).
        self._data_dict_ = {'max_progress': 0.0,
                            'current_progress': 0.0,
                            'prev_progress': 0.0,
                            'steps': 0.0,
                            'start_ndist': start_ndist,
                            'prev_car_pose': 0.0}

        #Load the action space
        self._action_space_, self._json_actions_ = \
            load_action_space(config_dict[const.ConfigParams.ACTION_SPACE_PATH.value])
        #! TODO evaluate if this is the best way to reset the car
        # Adding the reward data publisher
        self.reward_data_pub = RewardDataPublisher(self._agent_name_, self._json_actions_)
        # subscriber to time to update camera position
        self.camera_manager = CameraManager.get_instance()
        # True if the agent is in the training phase
        self._is_training_ = False
        # Register to the phase sink
        run_phase_sink.register(self)
        # Make sure velocity and angle are set to 0
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)

        # start_dist should be hypothetical start line (start_ndist) plus
        # start position offset (start_line_ndist_offset).
        start_pose = self._start_lane_.interpolate_pose(
            (self._data_dict_['start_ndist'] + self._start_line_ndist_offset) * self._track_data_.get_track_length(),
            finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        self._track_data_.initialize_object(self._agent_name_, start_pose, \
                                            ObstacleDimensions.BOT_CAR_DIMENSION)

        self.make_link_points = lambda link_state: Point(link_state.pose.position.x,
                                                         link_state.pose.position.y)
        self.reference_frames = ['' for _ in self._agent_link_name_list_]

        self._pause_car_model_pose = None
        self._park_position = DEFAULT_PARK_POSITION
        AbstractTracker.__init__(self, TrackerPriority.HIGH)

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        if self._pause_duration > 0.0:
            self._pause_duration -= delta_time

    @property
    def action_space(self):
        return self._action_space_

    def reset_agent(self):
        '''reset agent by reseting member variables, reset s3 metrics, and reset agent to
           starting position at the beginning of each episode
        '''
        logger.info("Reset agent")
        self._clear_data()
        self._metrics.reset()
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        start_model_state = self._get_car_start_model_state()
        # set_model_state and get_model_state is actually occurred asynchronously
        # in tracker with simulation clock subscription. So, when the agent is
        # entering next step function call, either set_model_state
        # or get_model_state may not actually happened and the agent position may be outdated.
        # To avoid such case, use blocking to actually update the model position in gazebo
        # and GetModelstateTracker to reflect the latest agent position right away when start.
        SetModelStateTracker.get_instance().set_model_state(start_model_state, blocking=True)
        GetModelStateTracker.get_instance().get_model_state(self._agent_name_, '', blocking=True)
        # reset view cameras
        self.camera_manager.reset(car_pose=start_model_state.pose,
                                  namespace=self._agent_name_)
        self._track_data_.update_object_pose(self._agent_name_, start_model_state.pose)

    def _pause_car_model(self, should_reset_camera=False):
        '''Pause agent immediately at the current position
        '''
        car_model_state = ModelState()
        car_model_state.model_name = self._agent_name_
        car_model_state.pose = self._pause_car_model_pose
        car_model_state.twist.linear.x = 0
        car_model_state.twist.linear.y = 0
        car_model_state.twist.linear.z = 0
        car_model_state.twist.angular.x = 0
        car_model_state.twist.angular.y = 0
        car_model_state.twist.angular.z = 0
        SetModelStateTracker.get_instance().set_model_state(car_model_state)
        if should_reset_camera:
            self.camera_manager.reset(car_pose=car_model_state.pose,
                                      namespace=self._agent_name_)

    def _park_car_model(self):
        '''Park agent after racer complete F1 race
        '''
        car_model_state = ModelState()
        car_model_state.model_name = self._agent_name_
        yaw = 0.0 if self._track_data_.is_ccw else math.pi
        orientation = euler_to_quaternion(yaw=yaw)
        car_model_state.pose.position.x = self._park_position[0]
        car_model_state.pose.position.y = self._park_position[1]
        car_model_state.pose.position.z = 0.0
        car_model_state.pose.orientation.x = orientation[0]
        car_model_state.pose.orientation.y = orientation[1]
        car_model_state.pose.orientation.z = orientation[2]
        car_model_state.pose.orientation.w = orientation[3]
        car_model_state.twist.linear.x = 0
        car_model_state.twist.linear.y = 0
        car_model_state.twist.linear.z = 0
        car_model_state.twist.angular.x = 0
        car_model_state.twist.angular.y = 0
        car_model_state.twist.angular.z = 0
        SetModelStateTracker.get_instance().set_model_state(car_model_state)
        self.camera_manager.reset(car_pose=car_model_state.pose,
                                  namespace=self._agent_name_)

    def _get_closest_obj(self, start_dist, name_filter=None):
        '''get the closest object dist and pose both ahead and behind

        Args:
            start_dist (float): start distance
            name_filter (str): name to filter for the closest object check

        Returns:
            tuple (float, ModelStates.pose): tuple of closest object distance and closest
                                             object pose
        '''
        closest_object_dist = None
        closest_object_pose = None
        closest_obj_gap = const.CLOSEST_OBJ_GAP
        for object_name, object_pose in self._track_data_.object_poses.items():
            if object_name != self._agent_name_:
                if name_filter and name_filter not in object_name:
                    continue
                object_point = Point([object_pose.position.x, object_pose.position.y])
                object_dist = self._track_data_.center_line.project(object_point)
                abs_object_gap = abs(object_dist - start_dist) % \
                                 self._track_data_.get_track_length()
                if abs_object_gap < closest_obj_gap:
                    closest_obj_gap = abs_object_gap
                    closest_object_dist = object_dist
                    closest_object_pose = object_pose
        return closest_object_dist, closest_object_pose

    def _get_reset_poses(self, dist):
        """
        Return center, outer, inner reset position based on given dist

        Args:
            dist(float): interpolated track dist

        Returns: tuple of center, outer, and inner rest positions

        """
        # It is extremely important to get the interpolated pose of cur_dist
        # using center line first. And then use the center line pose to
        # interpolate the inner and outer reset pose.
        # If cur_dist is directly used with inner lane and outer lane pose
        # interpolation then the result's pose difference from actual reset pose (where it should be)
        # is too large.
        cur_center_pose = self._track_data_.center_line.interpolate_pose(
            dist,
            finite_difference=FiniteDifference.FORWARD_DIFFERENCE)

        inner_reset_pose = self._track_data_.inner_lane.interpolate_pose(
            self._track_data_.inner_lane.project(Point(cur_center_pose.position.x,
                                                       cur_center_pose.position.y)),
            finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        outer_reset_pose = self._track_data_.outer_lane.interpolate_pose(
            self._track_data_.outer_lane.project(Point(cur_center_pose.position.x,
                                                       cur_center_pose.position.y)),
            finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        return cur_center_pose, inner_reset_pose, outer_reset_pose

    def _is_obstacle_inner(self, obstacle_pose):
        """Return whether given object is in inner lane.

        Args:
            obstacle_pose (Pose): Obstacle pose object

        Returns:
            bool: True for inner. False otherwise
        """
        obstacle_point = Point([obstacle_pose.position.x, obstacle_pose.position.y])
        obstacle_nearest_pnts_dict = self._track_data_.get_nearest_points(obstacle_point)
        obstacle_nearest_dist_dict = self._track_data_.get_nearest_dist(obstacle_nearest_pnts_dict,
                                                                        obstacle_point)
        return obstacle_nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value] < \
               obstacle_nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]

    def _get_car_reset_model_state(self, car_pose):
        '''Get car reset model state when car goes offtrack or crash into a static obstacle

        Args:
            car_pose (Pose): current car pose

        Returns:
            ModelState: reset state
        '''
        cur_dist = self._data_dict_['current_progress'] * \
                     self._track_data_.get_track_length() / 100.0
        closest_object_dist, closest_obstacle_pose = self._get_closest_obj(cur_dist, const.OBSTACLE_NAME_PREFIX)
        if closest_obstacle_pose is not None:
            # If static obstacle is in circumference of reset position,
            # put the car to opposite lane and 1m back.
            cur_dist = closest_object_dist - const.RESET_BEHIND_DIST
            cur_center_pose, inner_reset_pose, outer_reset_pose = self._get_reset_poses(dist=cur_dist)
            is_object_inner = self._is_obstacle_inner(obstacle_pose=closest_obstacle_pose)
            new_pose = outer_reset_pose if is_object_inner else inner_reset_pose
        else:
            cur_center_pose, inner_reset_pose, outer_reset_pose = self._get_reset_poses(dist=cur_dist)
            # If there is no obstacle interfering reset position, then
            # put the car back to closest lane from the off-track position.
            inner_distance = pose_distance(inner_reset_pose, car_pose)
            outer_distance = pose_distance(outer_reset_pose, car_pose)
            new_pose = inner_reset_pose if inner_distance < outer_distance else outer_reset_pose

        # check for whether reset pose is valid or not
        new_pose = self._check_for_invalid_reset_pose(pose=new_pose, dist=cur_dist)

        car_model_state = ModelState()
        car_model_state.model_name = self._agent_name_
        car_model_state.pose = new_pose
        car_model_state.twist.linear.x = 0
        car_model_state.twist.linear.y = 0
        car_model_state.twist.linear.z = 0
        car_model_state.twist.angular.x = 0
        car_model_state.twist.angular.y = 0
        car_model_state.twist.angular.z = 0
        return car_model_state

    def _get_car_start_model_state(self):
        '''Get car start model state. For training, if start position has an object,
           reset to the opposite lane. We assume that during training, there are no objects
           at both lane in the same progress. For evaluation, always start at progress 0.

        Returns:
            ModelState: start state
        '''
        # start_dist should be hypothetical start line (start_ndist) plus
        # start position offset (start_line_ndist_offset).
        start_dist = (self._data_dict_['start_ndist'] + self._start_line_ndist_offset) * self._track_data_.get_track_length()

        if self._is_training_:
            _, closest_object_pose = self._get_closest_obj(start_dist)
            # Compute the start pose based on start distance
            start_pose = self._track_data_.center_line.interpolate_pose(
                start_dist,
                finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
            # If closest_object_pose is not None, for example bot car is around agent
            # start position. The below logic checks for whether inner or outer lane
            # is available for placement. Then, it updates start_pose accordingly.
            if closest_object_pose is not None:
                object_point = Point([closest_object_pose.position.x, closest_object_pose.position.y])
                object_nearest_pnts_dict = self._track_data_.get_nearest_points(object_point)
                object_nearest_dist_dict = self._track_data_.get_nearest_dist(object_nearest_pnts_dict,
                                                                              object_point)
                object_is_inner = object_nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value] < \
                                  object_nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]
                if object_is_inner:
                    start_pose = self._track_data_.outer_lane.interpolate_pose(
                        self._track_data_.outer_lane.project(Point(start_pose.position.x,
                                                                   start_pose.position.y)),
                        finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
                else:
                    start_pose = self._track_data_.inner_lane.interpolate_pose(
                        self._track_data_.inner_lane.project(Point(start_pose.position.x,
                                                                   start_pose.position.y)),
                        finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        else:
            start_pose = self._start_lane_.interpolate_pose(
                start_dist,
                finite_difference=FiniteDifference.FORWARD_DIFFERENCE)

        # check for whether reset pose is valid or not
        start_pose = self._check_for_invalid_reset_pose(pose=start_pose, dist=start_dist)

        car_model_state = ModelState()
        car_model_state.model_name = self._agent_name_
        car_model_state.pose = start_pose
        car_model_state.twist.linear.x = 0
        car_model_state.twist.linear.y = 0
        car_model_state.twist.linear.z = 0
        car_model_state.twist.angular.x = 0
        car_model_state.twist.angular.y = 0
        car_model_state.twist.angular.z = 0
        return car_model_state


    def _check_for_invalid_reset_pose(self, pose, dist):
        # if current reset position/orientation is inf/-inf or nan, reset to the starting position centerline
        pose_list = [pose.position.x, pose.position.y, pose.position.z,
                     pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        if math.inf in pose_list or -math.inf in pose_list or math.nan in pose_list:
            logger.info("invalid reset pose {} for distance {}".format(pose_list, dist))
            pose, _, _ = self._get_reset_poses(dist=0.0)
            # if is training job, update to start_ndist to 0.0
            if self._is_training_:
                self._data_dict_['start_ndist'] = 0.0
        return pose

    def send_action(self, action):
        '''Publish action topic to gazebo to render

        Args:
            action (int): model metadata action_space index

        Raises:
            GenericRolloutException: Agent phase is not defined
        '''
        if self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.RUN.value:
            steering_angle = float(self._json_actions_[action]['steering_angle']) * math.pi / 180.0
            speed = float(self._json_actions_[action]['speed'] / const.WHEEL_RADIUS) \
                    * self._speed_scale_factor_
            send_action(self._velocity_pub_dict_, self._steering_pub_dict_, steering_angle, speed)
        elif self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PAUSE.value or \
                self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PARK.value:
            send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        else:
            raise GenericRolloutException('Agent phase {} is not defined'.\
                  format(self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value]))

    def _get_agent_pos(self, car_pose, car_link_points, relative_pos):
        '''Returns a dictionary with the keys defined in AgentPos which contains
           the position of the agent on the track, the location of the desired
           links, and the orientation of the agent.
           car_pose - Gazebo Pose of the agent
           car_link_points (Point[]) - List of car's links' Points.
           relative_pos - List containing the x-y relative position of the front of
                          the agent
        '''
        try:
            # Compute the model's orientation
            model_orientation = np.array([car_pose.orientation.x,
                                          car_pose.orientation.y,
                                          car_pose.orientation.z,
                                          car_pose.orientation.w])
            # Compute the model's location relative to the front of the agent
            model_location = np.array([car_pose.position.x,
                                       car_pose.position.y,
                                       car_pose.position.z]) + \
                             apply_orientation(model_orientation, np.array(relative_pos))
            model_point = Point(model_location[0], model_location[1])

            return {AgentPos.ORIENTATION.value: model_orientation,
                    AgentPos.POINT.value: model_point,
                    AgentPos.LINK_POINTS.value: car_link_points}
        except Exception as ex:
            raise GenericRolloutException("Unable to get position: {}".format(ex))

    def update_agent(self, action):
        '''Update agent reward and metrics ater the action is taken

        Args:
            action (int): model metadata action_space index

        Returns:
            dict: dictionary of agent info with agent name as the key and info as the value

        Raises:
            GenericRolloutException: Cannot find position
        '''
        # get car state
        car_model_state = GetModelStateTracker.get_instance().get_model_state(self._agent_name_, '')
        self._track_data_.update_object_pose(self._agent_name_, car_model_state.pose)
        link_states = [GetLinkStateTracker.get_instance().get_link_state(link_name, reference_frame).link_state
                       for link_name, reference_frame in zip(self._agent_link_name_list_, self.reference_frames)]
        link_points = [self.make_link_points(link_state) for link_state in link_states]

        current_car_pose = car_model_state.pose
        try:
            # Get the position of the agent
            pos_dict = self._get_agent_pos(current_car_pose,
                                           link_points,
                                           const.RELATIVE_POSITION_OF_FRONT_OF_CAR)
            model_point = pos_dict[AgentPos.POINT.value]
            self._data_dict_['steps'] += 1
        except Exception as ex:
            raise GenericRolloutException('Cannot find position: {}'.format(ex))
        # Set the reward and training metrics
        set_reward_and_metrics(self._reward_params_, self._step_metrics_,
                               self._agent_name_, pos_dict, self._track_data_,
                               self._data_dict_, action, self._json_actions_,
                               current_car_pose)
        prev_pnt_dist = min(model_point.distance(self._prev_waypoints_['prev_point']),
                            model_point.distance(self._prev_waypoints_['prev_point_2']))
        self._data_dict_['current_progress'] = self._reward_params_[const.RewardParam.PROG.value[0]]
        self._data_dict_['max_progress'] = max(self._data_dict_['max_progress'],
                                               self._data_dict_['current_progress'])
        self._prev_waypoints_['prev_point_2'] = self._prev_waypoints_['prev_point']
        self._prev_waypoints_['prev_point'] = model_point
        self._ctrl_status[AgentCtrlStatus.POS_DICT.value] = pos_dict
        self._ctrl_status[AgentCtrlStatus.STEPS.value] = self._data_dict_['steps']
        self._ctrl_status[AgentCtrlStatus.CURRENT_PROGRESS.value] = self._data_dict_['current_progress']
        self._ctrl_status[AgentCtrlStatus.PREV_PROGRESS.value] = self._data_dict_['prev_progress']
        self._ctrl_status[AgentCtrlStatus.PREV_PNT_DIST.value] = prev_pnt_dist
        self._ctrl_status[AgentCtrlStatus.START_NDIST.value] = self._data_dict_['start_ndist']
        return {self._agent_name_: self._reset_rules_manager.update(self._ctrl_status)}

    def judge_action(self, agents_info_map):
        '''Judge the action that agent just take

        Args:
            agents_info_map: Dictionary contains all agents info with agent name as the key
                             and info as the value

        Returns:
            tuple (float, bool, dict): tuple of reward, done flag, and step metrics

        Raises:
            RewardFunctionError: Reward function exception
            GenericRolloutException: reward is nan or inf
        '''
        # check agent status to update reward and done flag
        reset_rules_status = self._reset_rules_manager.get_dones()
        self._reward_params_[const.RewardParam.CRASHED.value[0]] = \
            reset_rules_status[EpisodeStatus.CRASHED.value]
        self._reward_params_[const.RewardParam.OFFTRACK.value[0]] = \
            reset_rules_status[EpisodeStatus.OFF_TRACK.value]
        episode_status, pause, done = self._check_for_episode_termination(reset_rules_status, agents_info_map)
        if not pause and not done:
            # If episode termination check returns status as not paused and not done, and
            # if reset_rules_status's CRASHED is true, then the crashed object must have smaller normalize progress
            # compare to rollout agent.
            # - if reset_rules_status's CRASHED is false, then reward params' CRASHED should be already false.
            # In such case, from rollout_agent's perspective, it should consider it as there is no crash.
            # Therefore, setting reward params' CRASHED as false if not paused and not done.
            self._reward_params_[const.RewardParam.CRASHED.value[0]] = False
        if self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.RUN.value:
            reward = self._judge_action_at_run_phase(episode_status=episode_status, pause=pause)
        elif self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PAUSE.value:
            reward, episode_status = self._judge_action_at_pause_phase(episode_status=episode_status, done=done)
        elif self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PARK.value:
            self._park_car_model()
            episode_status, pause, done = EpisodeStatus.PARK.value, False, True
            reward = const.ZERO_REWARD
        else:
            raise GenericRolloutException('Agent phase {} is not defined'.\
                  format(self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value]))
        # update and upload metrics
        self._step_metrics_[StepMetrics.REWARD.value] = reward
        self._step_metrics_[StepMetrics.DONE.value] = done
        self._step_metrics_[StepMetrics.TIME.value] = time.time()
        self._step_metrics_[StepMetrics.EPISODE_STATUS.value] = episode_status
        self._data_dict_['prev_progress'] = 0.0 if self._step_metrics_[StepMetrics.PROG.value] == 100 \
                                                else self._step_metrics_[StepMetrics.PROG.value]
        if self._data_dict_['current_progress'] == 100:
             self._data_dict_['max_progress'] = 0.0
             self._data_dict_['current_progress'] = 0.0
        self._metrics.upload_step_metrics(self._step_metrics_)
        if self._is_continuous and self._reward_params_[const.RewardParam.PROG.value[0]] == 100:
            self._metrics.append_episode_metrics()
            self._metrics.reset()
            self._reset_rules_manager.reset()
        return reward, done, self._step_metrics_

    def _judge_action_at_run_phase(self, episode_status, pause):
        self._pause_duration = 0.0
        current_car_pose = self._track_data_.get_object_pose(self._agent_name_)
        try:
            reward = float(self._reward_(copy.deepcopy(self._reward_params_)))
        except Exception as ex:
            raise RewardFunctionError('Reward function exception {}'.format(ex))
        if math.isnan(reward) or math.isinf(reward):
            raise RewardFunctionError('{} returned as reward'.format(reward))
        # transition to AgentPhase.PARK.value when episode complete and done condition is all
        if episode_status == EpisodeStatus.EPISODE_COMPLETE.value and \
                self._done_condition == all:
            self._park_position = self._track_data_.pop_park_position()
            self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.PARK.value
            self._park_car_model()
        # transition to AgentPhase.PAUSE.value
        if pause:
            should_reset_camera = False
            pause_car_model_pose = current_car_pose
            # add pause time based on different paused status
            if episode_status == EpisodeStatus.CRASHED.value:
                self._pause_duration += self._collision_penalty
                # add blink effect and remove current agent from collision list
                if self._collision_penalty > 0.0:
                    effect = BlinkEffect(model_name=self._agent_name_,
                                         min_alpha=const.BLINK_MIN_ALPHA,
                                         interval=const.BLINK_INTERVAL,
                                         duration=self._collision_penalty)
                    effect.attach()
                # If crash into an static obstacle, reset first and then pause. This will prevent
                # agent and obstacle wiggling around because bit mask is not used between agent
                # and static obstacle
                if 'obstacle' in self._curr_crashed_object_name:
                    pause_car_model_pose = self._get_car_reset_model_state(
                        car_pose=current_car_pose).pose
                    should_reset_camera = True
            elif episode_status == EpisodeStatus.OFF_TRACK.value:
                self._pause_duration += self._off_track_penalty
                # add blink effect and remove current agent from collision list
                if self._off_track_penalty > 0.0:
                    effect = BlinkEffect(model_name=self._agent_name_,
                                         min_alpha=const.BLINK_MIN_ALPHA,
                                         interval=const.BLINK_INTERVAL,
                                         duration=self._off_track_penalty)
                    effect.attach()
                    # when agent off track current car pose might be closer
                    # to other part of the track. Therefore, instead of using
                    # current car pose to calculate reset position, the previous
                    # car pose is used.
                    pause_car_model_pose = self._get_car_reset_model_state(
                        car_pose=self._data_dict_['prev_car_pose']).pose
                    should_reset_camera = True
            self._pause_car_model_pose = pause_car_model_pose
            self._pause_car_model(should_reset_camera=should_reset_camera)
            self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.PAUSE.value
        self._data_dict_['prev_car_pose'] = current_car_pose
        return reward

    def _judge_action_at_pause_phase(self, episode_status, done):
        reward = const.ZERO_REWARD
        self._pause_car_model()
        # transition to AgentPhase.RUN.value
        if self._pause_duration <= 0.0:
            # if reset during pause, do not reset again after penalty seconds is over
            self._reset_rules_manager.reset()
            self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.RUN.value
        if not done:
            episode_status = EpisodeStatus.PAUSE.value
        return reward, episode_status

    def _check_for_episode_termination(self, reset_rules_status, agents_info_map):
        '''Check for whether a episode should be terminated

        Args:
            reset_rules_status: dictionary of reset rules status with key as reset rule names and value as
                                reset rule bool status
            agents_info_map: dictionary of agents info map with key as agent name and value as agent info

        Returns:
            tuple (string, bool, bool): episode status, pause flag, and done flag
        '''
        episode_status = EpisodeStatus.get_episode_status(reset_rules_status)
        pause = False
        done = False
        # Note: check EPISODE_COMPLETE as the first item because agent might crash
        # at the finish line.
        if EpisodeStatus.EPISODE_COMPLETE.value in reset_rules_status and \
                reset_rules_status[EpisodeStatus.EPISODE_COMPLETE.value]:
            done = True
            episode_status = EpisodeStatus.EPISODE_COMPLETE.value
        elif EpisodeStatus.CRASHED.value in reset_rules_status and \
                reset_rules_status[EpisodeStatus.CRASHED.value]:
            # only check for crash when at RUN phase
            if self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.RUN.value:
                self._curr_crashed_object_name = agents_info_map[self._agent_name_][AgentInfo.CRASHED_OBJECT_NAME.value]
                # check crash with all other objects besides static obstacle
                if 'obstacle' not in self._curr_crashed_object_name:
                    current_progress = agents_info_map[self._agent_name_][AgentInfo.CURRENT_PROGRESS.value]
                    crashed_obj_info = agents_info_map[self._curr_crashed_object_name]
                    crashed_obj_progress = crashed_obj_info[AgentInfo.CURRENT_PROGRESS.value]
                    crashed_obj_start_ndist = crashed_obj_info[AgentInfo.START_NDIST.value]
                    crashed_object_progress = get_normalized_progress(crashed_obj_progress,
                                                                      start_ndist=crashed_obj_start_ndist)
                    current_progress = get_normalized_progress(current_progress,
                                                               start_ndist=self._data_dict_['start_ndist'])
                    if current_progress < crashed_object_progress:
                        done, pause = self._check_for_phase_change()
                    else:
                        episode_status = EpisodeStatus.IN_PROGRESS.value
                else:
                    done, pause = self._check_for_phase_change()
            else:
                pause = True
        elif any(reset_rules_status.values()):
            done, pause = self._check_for_phase_change()
        return episode_status, pause, done

    def _check_for_phase_change(self):
        '''check whether to pause a agent

        Returns:
            tuple(bool, bool): done flag and pause flag
        '''
        done, pause = True, False
        if self._reset_count < self._number_of_resets:
            self._reset_count += 1
            self._reset_rules_manager.reset()
            done, pause = False, True
        return done, pause

    def finish_episode(self):
        '''finish episode by appending episode metrics, upload metrics, and alternate direction
        if needed
        '''
        if not self._is_continuous:
            self._metrics.append_episode_metrics()
        self._metrics.upload_episode_metrics()
        if self._start_pos_behavior_['change_start'] and self._is_training_:
            self._data_dict_['start_ndist'] = (self._data_dict_['start_ndist']
                                               + const.ROUND_ROBIN_ADVANCE_DIST) % 1.0
        # For multi-agent case, alternating direction will NOT work!
        # Reverse direction will be set multiple times
        # However, we are not supporting multi-agent training for now
        if self._start_pos_behavior_['alternate_dir'] and self._is_training_:
            self._track_data_.reverse_dir = not self._track_data_.reverse_dir

    def _clear_data(self):
        '''clear data at the beginning of a new episode
        '''
        self._curr_crashed_object_name = ''
        self._reset_count = 0
        self._reset_rules_manager.reset()
        self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.RUN.value
        for key in self._prev_waypoints_:
            self._prev_waypoints_[key] = Point(0, 0)
        for key in self._data_dict_:
            if key != 'start_ndist':
                self._data_dict_[key] = 0.0

    def update(self, data):
        self._is_training_ = data == RunPhase.TRAIN
