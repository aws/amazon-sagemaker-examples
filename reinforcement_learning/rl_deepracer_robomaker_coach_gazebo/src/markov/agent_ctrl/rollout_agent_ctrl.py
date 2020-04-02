'''This module implements concrete agent controllers for the rollout worker'''
import copy
import time
from collections import OrderedDict
import math
import rospy
import threading
import logging

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from shapely.geometry import Point
from markov.visualizations.reward_distributions import RewardDataPublisher

import markov.agent_ctrl.constants as const
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.agent_ctrl.utils import set_reward_and_metrics, \
                                    send_action, load_action_space, get_speed_factor, \
                                    Logger
from markov.track_geom.constants import AgentPos, TrackNearDist, SET_MODEL_STATE, \
                                        GET_MODEL_STATE, ObstacleDimensions
from markov.track_geom.track_data import FiniteDifference, TrackData
from markov.metrics.constants import StepMetrics, EpisodeStatus
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.cameras.camera_manager import CameraManager
from markov.common import ObserverInterface
from markov.deepracer_exceptions import RewardFunctionError, GenericRolloutException
from markov.reset.constants import AgentPhase, AgentCtrlStatus, AgentInfo
from markov.reset.reset_rules_manager import ResetRulesManager
from markov.reset.utils import construct_reset_rules_manager
from rl_coach.core_types import RunPhase

logger = Logger(__name__, logging.INFO).get_logger()

class RolloutCtrl(AgentCtrlInterface, ObserverInterface):
    '''Concrete class for an agent that drives forward'''
    def __init__(self, config_dict, run_phase_sink, metrics):
        '''agent_name - String containing the name of the agent
           config_dict - Dictionary containing all the keys in ConfigParams
           run_phase_sink - Sink to recieve notification of a change in run phase
        '''
        # reset rules manager
        self._metrics = metrics
        self._is_continuous = config_dict[const.ConfigParams.IS_CONTINUOUS.value]
        self._is_reset = False
        self._reset_rules_manager = construct_reset_rules_manager(config_dict)
        self._ctrl_status = dict()
        self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.RUN.value
        self._config_dict = config_dict
        self._number_of_resets = config_dict[const.ConfigParams.NUMBER_OF_RESETS.value]
        self._off_track_penalty = config_dict[const.ConfigParams.OFF_TRACK_PENALTY.value]
        self._collision_penalty = config_dict[const.ConfigParams.COLLISION_PENALTY.value]
        self._pause_end_time = 0.0
        self._reset_count = 0
        self._curr_crashed_object_name = None
        self._last_crashed_object_name = None
        # simapp_version speed scale
        self._speed_scale_factor_ = get_speed_factor(config_dict[const.ConfigParams.VERSION.value])
        # Store the name of the agent used to set agents position on the track
        self._agent_name_ = config_dict[const.ConfigParams.AGENT_NAME.value]
        # Store the name of the links in the agent, this should be const
        self._agent_link_name_list_ = config_dict[const.ConfigParams.LINK_NAME_LIST.value]
        # Store the reward function
        self._reward_ = config_dict[const.ConfigParams.REWARD.value]
        self._track_data_ = TrackData.get_instance()
        # Create publishers for controlling the car
        self._velocity_pub_dict_ = OrderedDict()
        self._steering_pub_dict_ = OrderedDict()
        for topic in config_dict[const.ConfigParams.VELOCITY_LIST.value]:
            self._velocity_pub_dict_[topic] = rospy.Publisher(topic, Float64, queue_size=1)
        for topic in config_dict[const.ConfigParams.STEERING_LIST.value]:
            self._steering_pub_dict_[topic] = rospy.Publisher(topic, Float64, queue_size=1)
        #Create default reward parameters
        self._reward_params_ = const.RewardParam.make_default_param()
        #Creat the default metrics dictionary
        self._step_metrics_ = StepMetrics.make_default_metric()
        # State variable to track if the car direction has been reversed
        self._reverse_dir_ = False
        # Dictionary of bools indicating starting position behavior
        self._start_pos_behavior_ = \
            {'change_start' : config_dict[const.ConfigParams.CHANGE_START.value],
             'alternate_dir' : config_dict[const.ConfigParams.ALT_DIR.value]}
        # Dictionary to track the previous way points
        self._prev_waypoints_ = {'prev_point' : Point(0, 0), 'prev_point_2' : Point(0, 0)}
        # Dictionary containing some of the data for the agent
        self._data_dict_ = {'max_progress': 0.0,
                            'current_progress': 0.0,
                            'prev_progress': 0.0,
                            'steps': 0.0,
                            'start_ndist': 0.0}
        #Load the action space
        self._action_space_, self._json_actions_ = \
            load_action_space(config_dict[const.ConfigParams.ACTION_SPACE_PATH.value])
        #! TODO evaluate if this is the best way to reset the car
        rospy.wait_for_service(SET_MODEL_STATE)
        rospy.wait_for_service(GET_MODEL_STATE)
        self.set_model_state = ServiceProxyWrapper(SET_MODEL_STATE, SetModelState)
        self.get_model_client = ServiceProxyWrapper(GET_MODEL_STATE, GetModelState)
        # Adding the reward data publisher
        self.reward_data_pub = RewardDataPublisher(self._agent_name_, self._json_actions_)
        # init time
        self.last_time = 0.0
        self.curr_time = 0.0
        # subscriber to time to update camera position
        self.camera_manager = CameraManager.get_instance()
        # True if the agent is in the training phase
        self._is_training_ = False
        rospy.Subscriber('/clock', Clock, self._update_sim_time)
        # Register to the phase sink
        run_phase_sink.register(self)
        # Make sure velicty and angle are set to 0
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        start_pose = self._track_data_._center_line_.interpolate_pose(self._data_dict_['start_ndist'] * self._track_data_.get_track_length(),
                                                                      reverse_dir=self._reverse_dir_,
                                                                      finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        self._track_data_.initialize_object(self._agent_name_, start_pose, ObstacleDimensions.BOT_CAR_DIMENSION)
        self.car_model_state = self.get_model_client(self._agent_name_, '')
        self._reset_agent(reset_pos=const.ResetPos.START_POS.value)

    def _update_sim_time(self, sim_time):
        '''Callback to rospy clock to update time, camera, and penalty second

        Args:
            sim_time (float): current gazebo simulation time
        '''
        # update timer
        self.curr_time = sim_time.clock.secs + 1.e-9*sim_time.clock.nsecs
        delta_time = self.curr_time - self.last_time
        self.last_time = self.curr_time
        # get car state
        self.car_model_state = self.get_model_client(self._agent_name_, '')
        self.update_camera(state=self.car_model_state,
                           delta_time=delta_time,
                           namespace=self._agent_name_)

    def update_camera(self, state, delta_time, namespace):
        '''Update camera pose
        '''
        self.camera_manager.update(state=state,
                                   delta_time=delta_time,
                                   namespace=namespace)

    @property
    def action_space(self):
        return self._action_space_

    def reset_agent(self):
        '''reset agent by reseting member variables, reset s3 metrics, and reset agent to
           starting position
        '''
        self._clear_data()
        self._metrics.reset()
        self._reset_agent(reset_pos=const.ResetPos.START_POS.value)

    def _reset_agent(self, reset_pos):
        '''Reset agent to either starting pos or last pos

        Args:
            reset_pos (string): start_pos/last_pos depending on reset
                                to starting position of the lap or position
                                from last frame

        Raises:
            GenericRolloutException: Reset position is not defined
        '''
        logger.info("Reset agent")
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        if reset_pos == const.ResetPos.LAST_POS.value:
            self._track_data_.car_ndist = self._data_dict_['current_progress']
            start_dist = self._data_dict_['current_progress'] * \
                         self._track_data_.get_track_length() / 100.0
            start_model_state = self._get_car_reset_model_state(start_dist)
        elif reset_pos == const.ResetPos.START_POS.value:
            self._track_data_.car_ndist = self._data_dict_['start_ndist']
            start_dist = self._data_dict_['start_ndist'] * self._track_data_.get_track_length()
            start_model_state = self._get_car_start_model_state(start_dist)
        else:
            raise GenericRolloutException('Reset position {} is not defined'.format(reset_pos))
        self.set_model_state(start_model_state)
        # reset view cameras
        self.camera_manager.reset(start_model_state, namespace=self._agent_name_)

    def _pause_car_model(self):
        '''Pause agent immediately at the current position
        '''
        car_model_state = ModelState()
        car_model_state.model_name = self._agent_name_
        car_model_state.pose = self.car_model_state.pose
        car_model_state.twist.linear.x = 0
        car_model_state.twist.linear.y = 0
        car_model_state.twist.linear.z = 0
        car_model_state.twist.angular.x = 0
        car_model_state.twist.angular.y = 0
        car_model_state.twist.angular.z = 0
        self.set_model_state(car_model_state)

    def _get_closest_obj(self, start_dist):
        '''get the closest object dist and pose both ahead and behind

        Args:
            start_dist (float): start distance

        Returns:
            tuple (float, ModelStates.pose): tuple of closest object distance and closest
                                             object pose
        '''
        closest_object_dist = None
        closest_object_pose = None
        closest_obj_gap = const.CLOSEST_OBJ_GAP
        for object_name, object_pose in self._track_data_.object_poses.items():
            if object_name != self._agent_name_:
                object_point = Point([object_pose.position.x, object_pose.position.y])
                object_dist = self._track_data_._center_line_.project(object_point)
                object_dist_ahead = (object_dist - start_dist) % self._track_data_.get_track_length()
                object_dist_behind = (start_dist - object_dist) % self._track_data_.get_track_length()
                if self._reverse_dir_:
                    object_dist_ahead, object_dist_behind = object_dist_behind, object_dist_ahead
                if min(object_dist_ahead, object_dist_behind) < closest_obj_gap:
                    closest_obj_gap = min(object_dist_ahead, object_dist_behind)
                    closest_object_dist = object_dist
                    closest_object_pose = object_pose
        return closest_object_dist, closest_object_pose

    def _get_car_reset_model_state(self, start_dist):
        '''get avaliable car reset model state when reset is allowed

        Args:
            start_dist (float): start distance

        Returns:
            ModelState: start state
        '''
        closest_object_dist, _ = self._get_closest_obj(start_dist)
        # Check to place behind car if needed
        if closest_object_dist is not None:
            start_dist = closest_object_dist - const.RESET_BEHIND_DIST
        # Compute the start pose
        start_pose = self._track_data_._center_line_.interpolate_pose(start_dist,
                                                                      reverse_dir=self._reverse_dir_,
                                                                      finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        start_model_state = ModelState()
        start_model_state.model_name = self._agent_name_
        start_model_state.pose = start_pose
        start_model_state.twist.linear.x = 0
        start_model_state.twist.linear.y = 0
        start_model_state.twist.linear.z = 0
        start_model_state.twist.angular.x = 0
        start_model_state.twist.angular.y = 0
        start_model_state.twist.angular.z = 0
        return start_model_state

    def _get_car_start_model_state(self, start_dist):
        '''get avaliable car start model state when starting a new lap

        Args:
            start_dist (float): start distance

        Returns:
            ModelState: start state
        '''
        _, closest_object_pose = self._get_closest_obj(start_dist)
        # Compute the start pose
        start_pose = self._track_data_._center_line_.interpolate_pose(start_dist,
                                                                      reverse_dir=self._reverse_dir_,
                                                                      finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        # Check to place in inner or outer lane
        if closest_object_pose is not None:
            object_point = Point([closest_object_pose.position.x, closest_object_pose.position.y])
            object_nearest_pnts_dict = self._track_data_.get_nearest_points(object_point)
            object_nearest_dist_dict = self._track_data_.get_nearest_dist(object_nearest_pnts_dict, object_point)
            object_is_inner = object_nearest_dist_dict[TrackNearDist.NEAR_DIST_IN.value] < \
                                object_nearest_dist_dict[TrackNearDist.NEAR_DIST_OUT.value]
            if object_is_inner:
                start_pose = self._track_data_._outer_lane_.interpolate_pose(
                    self._track_data_._outer_lane_.project(Point(start_pose.position.x, start_pose.position.y)),
                                                        reverse_dir=self._reverse_dir_,
                                                        finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
            else:
                start_pose = self._track_data_._inner_lane_.interpolate_pose(
                    self._track_data_._inner_lane_.project(Point(start_pose.position.x, start_pose.position.y)),
                                                        reverse_dir=self._reverse_dir_,
                                                        finite_difference=FiniteDifference.FORWARD_DIFFERENCE)
        start_model_state = ModelState()
        start_model_state.model_name = self._agent_name_
        start_model_state.pose = start_pose
        start_model_state.twist.linear.x = 0
        start_model_state.twist.linear.y = 0
        start_model_state.twist.linear.z = 0
        start_model_state.twist.angular.x = 0
        start_model_state.twist.angular.y = 0
        start_model_state.twist.angular.z = 0
        return start_model_state

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
        elif self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PAUSE.value:
            send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        else:
            raise GenericRolloutException('Agent phase {} is not defined'.\
                  format(self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value]))

    def update_agent(self, action):
        '''Update agent reward and metrics ater the action is taken

        Args:
            action (int): model metadata action_space index

        Returns:
            dict: dictionary of agent info with agent name as the key and info as the value

        Raises:
            GenericRolloutException: Cannot find position
        '''
        try:
            # Get the position of the agent
            pos_dict = self._track_data_.get_agent_pos(self._agent_name_,
                                                       self._agent_link_name_list_,
                                                       const.RELATIVE_POSITION_OF_FRONT_OF_CAR)
            model_point = pos_dict[AgentPos.POINT.value]
            self._data_dict_['steps'] += 1
        except Exception as ex:
            raise GenericRolloutException('Cannot find position: {}'.format(ex))
        # Set the reward and training metrics
        set_reward_and_metrics(self._reward_params_, self._step_metrics_,
                               self._agent_name_, pos_dict, self._track_data_,
                               self._reverse_dir_, self._data_dict_, action, self._json_actions_)
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
        return {self._agent_name_ : self._reset_rules_manager.update(self._ctrl_status)}

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
        if self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.RUN.value:
            self._pause_end_time = self.curr_time
            try:
                reward = float(self._reward_(copy.deepcopy(self._reward_params_)))
            except Exception as ex:
                raise RewardFunctionError('Reward function exception {}'.format(ex))
            if math.isnan(reward) or math.isinf(reward):
                raise RewardFunctionError('{} returned as reward'.format(reward))
            # transition to AgentPhase.PAUSE.value
            if pause:
                # add pause time based on different paused status
                if episode_status == EpisodeStatus.CRASHED.value:
                    self._pause_end_time += self._collision_penalty
                elif episode_status == EpisodeStatus.OFF_TRACK.value:
                    self._pause_end_time += self._off_track_penalty
                self._pause_car_model()
                self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.PAUSE.value
        elif self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] == AgentPhase.PAUSE.value:
            reward = const.PAUSE_REWARD
            # use pause count to skip crash check for first N frames to prevent the first pause
            # frame agent position is too close to last crash position
            if episode_status == EpisodeStatus.CRASHED.value and \
                    self._curr_crashed_object_name != self._last_crashed_object_name:
                self._reset_agent(const.ResetPos.LAST_POS.value)
                self._is_reset = True
            # transition to AgentPhase.RUN.value
            if self.curr_time > self._pause_end_time:
                # if reset during pause, do not reset again after penalty seconds is over
                if not self._is_reset:
                    self._reset_agent(const.ResetPos.LAST_POS.value)
                self._is_reset = False
                self._reset_rules_manager.reset()
                self._last_crashed_object_name = None
                self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value] = AgentPhase.RUN.value
            if not done:
                episode_status = EpisodeStatus.PAUSE.value
        else:
            raise GenericRolloutException('Agent phase {} is not defined'.\
                  format(self._ctrl_status[AgentCtrlStatus.AGENT_PHASE.value]))
        # metrics
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
            self._curr_crashed_object_name = agents_info_map[self._agent_name_][AgentInfo.CRASHED_OBJECT_NAME.value]
            # check crash with all other objects besides static obstacle
            if 'obstacle' not in self._curr_crashed_object_name:
                current_progress = agents_info_map[self._agent_name_][AgentInfo.CURRENT_PROGRESS.value]
                crashed_object_progress = agents_info_map[self._curr_crashed_object_name]\
                                              [AgentInfo.CURRENT_PROGRESS.value]
                if current_progress < crashed_object_progress:
                    if self._curr_crashed_object_name != self._last_crashed_object_name:
                        self._last_crashed_object_name = self._curr_crashed_object_name
                        done, pause = self._check_for_phase_change()
                else:
                    # rewrite episode status to in progress if not this agent's fault for crash
                    episode_status = EpisodeStatus.IN_PROGRESS.value
            else:
                done, pause = self._check_for_phase_change()
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
        if not self._is_continuous:
            self._metrics.append_episode_metrics()
        self._metrics.upload_episode_metrics()
        if self._start_pos_behavior_['change_start'] and self._is_training_:
            self._data_dict_['start_ndist'] = (self._data_dict_['start_ndist']
                                               + const.ROUND_ROBIN_ADVANCE_DIST) % 1.0
        if self._start_pos_behavior_['alternate_dir'] and self._is_training_:
            self._reverse_dir_ = not self._reverse_dir_

    def _clear_data(self):
        self._is_reset = False
        self._last_crashed_object_name = None
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
