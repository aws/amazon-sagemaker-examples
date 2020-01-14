'''This module implements concrete agent controllers for the rollout worker'''
import copy
import time
from collections import OrderedDict
import math
import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from shapely.geometry import Point
from markov.visualizations.reward_distributions import RewardDataPublisher

import markov.agent_ctrl.constants as const
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.agent_ctrl.utils import set_reward_and_metrics, compute_current_prog, \
                                    send_action, load_action_space, get_speed_factor
from markov.track_geom.constants import AgentPos, TrackNearDist, SET_MODEL_STATE, \
                                        GET_MODEL_STATE
from markov.track_geom.track_data import FiniteDifference, TrackData
from markov.metrics.constants import StepMetrics, EpisodeStatus
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.cameras.camera_manager import CameraManager
from markov.cameras.camera_factory import CameraType
from markov.deepracer_exceptions import RewardFunctionError, GenericRolloutException

class RolloutCtrl(AgentCtrlInterface):
    '''Concrete class for an agent that drives forward'''
    def __init__(self, config_dict):
        '''agent_name - String containing the name of the agent
           config_dict - Dictionary containing all the keys in ConfigParams
        '''
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
        self._data_dict_ = {'prev_progress': 0.0, 'steps' : 0.0, 'start_ndist' : 0.0}
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
        camera_types = [camera for camera in CameraType]
        self.camera_manager = CameraManager(camera_types=camera_types)
        rospy.Subscriber('/clock', Clock, self.update_camera)
        # Make sure velicty and angle are set to 0
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)

    def update_camera(self, sim_time):
        # get car state
        car_model_state = self.get_model_client('racecar', '')
        # update timer
        self.curr_time = sim_time.clock.secs + 1.e-9*sim_time.clock.nsecs
        time_delta = self.curr_time - self.last_time
        self.last_time = self.curr_time
        # update camera pose
        self.camera_manager.update(car_model_state, time_delta)

    @property
    def action_space(self):
        return self._action_space_

    def reset_agent(self):
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, 0.0, 0.0)
        self._track_data_.car_ndist = self._data_dict_['start_ndist'] # TODO -- REMOVE THIS

        # Compute the start pose
        start_dist = self._data_dict_['start_ndist'] * self._track_data_.get_track_length()
        start_pose = self._track_data_._center_line_.interpolate_pose(start_dist,
                                                                      reverse_dir=self._reverse_dir_,
                                                                      finite_difference=FiniteDifference.FORWARD_DIFFERENCE)

        # If we have obstacles, don't start near one
        for object_pose in self._track_data_.object_poses.values():
            object_point = Point([object_pose.position.x, object_pose.position.y])
            object_dist = self._track_data_._center_line_.project(object_point)
            object_dist_ahead = (object_dist - start_dist) % self._track_data_.get_track_length()
            object_dist_behind = (start_dist - object_dist) % self._track_data_.get_track_length()
            if self._reverse_dir_:
                object_dist_ahead, object_dist_behind = object_dist_behind, object_dist_ahead
            if object_dist_ahead < 1.0 or object_dist_behind < 0.5: # TODO: don't hard-code these numbers
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
                break

        start_state = ModelState()
        start_state.model_name = self._agent_name_
        start_state.pose = start_pose
        start_state.twist.linear.x = 0
        start_state.twist.linear.y = 0
        start_state.twist.linear.z = 0
        start_state.twist.angular.x = 0
        start_state.twist.angular.y = 0
        start_state.twist.angular.z = 0
        self.set_model_state(start_state)
        # reset view cameras
        self.camera_manager.reset(start_state)

    def send_action(self, action):
        steering_angle = float(self._json_actions_[action]['steering_angle']) * math.pi / 180.0
        speed = float(self._json_actions_[action]['speed']/const.WHEEL_RADIUS)\
                *self._speed_scale_factor_
        send_action(self._velocity_pub_dict_, self._steering_pub_dict_, steering_angle, speed)

    def judge_action(self, action):
        try:
            # Get the position of the agent
            pos_dict = self._track_data_.get_agent_pos(self._agent_name_,
                                                       self._agent_link_name_list_,
                                                       const.RELATIVE_POSITION_OF_FRONT_OF_CAR)
            model_point = pos_dict[AgentPos.POINT.value]
            # Compute the next index
            current_ndist = self._track_data_.get_norm_dist(model_point)
            prev_index, next_index = self._track_data_.find_prev_next_waypoints(current_ndist,
                                                                                normalized=True,
                                                                                reverse_dir=self._reverse_dir_)
            # Set the basic reward and training metrics
            set_reward_and_metrics(self._reward_params_, self._step_metrics_, pos_dict,
                                   self._track_data_, next_index, prev_index, action,
                                   self._json_actions_)
            # Convert current progress to be [0,100] starting at the initial waypoint
            if self._reverse_dir_:
                self._reward_params_[const.RewardParam.LEFT_CENT.value[0]] = \
                    not self._reward_params_[const.RewardParam.LEFT_CENT.value[0]]
                current_progress = self._data_dict_['start_ndist'] - current_ndist
            else:
                current_progress = current_ndist - self._data_dict_['start_ndist']

            current_progress = compute_current_prog(current_progress,
                                                    self._data_dict_['prev_progress'])
            self._data_dict_['steps'] += 1
            # Add the agen specific metrics
            self._step_metrics_[StepMetrics.STEPS.value] = \
            self._reward_params_[const.RewardParam.STEPS.value[0]] = self._data_dict_['steps']
            self._reward_params_[const.RewardParam.REVERSE.value[0]] = self._reverse_dir_
            self._step_metrics_[StepMetrics.PROG.value] = \
            self._reward_params_[const.RewardParam.PROG.value[0]] = current_progress
        except Exception as ex:
            raise GenericRolloutException('Cannot find position: {}'.format(ex))

        # This code should be replaced with the contact sensor code
        is_crashed = False
        model_heading = self._reward_params_[const.RewardParam.HEADING.value[0]]

        obstacle_reward_params = self._track_data_.get_object_reward_params(model_point, model_heading,
                                     current_progress, self._reverse_dir_)
        if obstacle_reward_params:
            self._reward_params_.update(obstacle_reward_params)
            is_crashed = self._track_data_.is_racecar_collided(pos_dict[AgentPos.LINK_POINTS.value])

        prev_pnt_dist = min(model_point.distance(self._prev_waypoints_['prev_point']),
                            model_point.distance(self._prev_waypoints_['prev_point_2']))

        is_off_track = not any(self._track_data_.points_on_track(pos_dict[AgentPos.LINK_POINTS.value]))
        is_immobilized = (prev_pnt_dist <= 0.0001 and self._data_dict_['steps'] % const.NUM_STEPS_TO_CHECK_STUCK == 0)
        is_lap_complete = current_progress >= 100.0

        self._reward_params_[const.RewardParam.CRASHED.value[0]] = is_crashed
        self._reward_params_[const.RewardParam.OFFTRACK.value[0]] = is_off_track

        done = is_crashed or is_immobilized or is_off_track or is_lap_complete
        episode_status = EpisodeStatus.get_episode_status(is_crashed=is_crashed,
                                                          is_immobilized=is_immobilized,
                                                          is_off_track=is_off_track,
                                                          is_lap_complete=is_lap_complete)
        try:
            reward = float(self._reward_(copy.deepcopy(self._reward_params_)))
        except Exception as ex:
            raise RewardFunctionError('Reward function exception {}'.format(ex))
        if math.isnan(reward) or math.isinf(reward):
            raise RewardFunctionError('{} returned as reward'.format(reward))

        self._prev_waypoints_['prev_point_2'] = self._prev_waypoints_['prev_point']
        self._prev_waypoints_['prev_point'] = model_point
        self._data_dict_['prev_progress'] = current_progress
        #Get the last of the step metrics
        self._step_metrics_[StepMetrics.REWARD.value] = reward
        self._step_metrics_[StepMetrics.DONE.value] = done
        self._step_metrics_[StepMetrics.TIME.value] = time.time()
        self._step_metrics_[StepMetrics.EPISODE_STATUS.value] = episode_status.value

        return reward, done, self._step_metrics_

    def finish_episode(self):
        if self._start_pos_behavior_['change_start']:
            self._data_dict_['start_ndist'] = (self._data_dict_['start_ndist']
                                               + const.ROUND_ROBIN_ADVANCE_DIST) % 1.0
        if self._start_pos_behavior_['alternate_dir']:
            self._reverse_dir_ = not self._reverse_dir_

    def clear_data(self):
        for key in self._prev_waypoints_:
            self._prev_waypoints_[key] = Point(0, 0)
        for key in self._data_dict_:
            if key != 'start_ndist':
                self._data_dict_[key] = 0.0
