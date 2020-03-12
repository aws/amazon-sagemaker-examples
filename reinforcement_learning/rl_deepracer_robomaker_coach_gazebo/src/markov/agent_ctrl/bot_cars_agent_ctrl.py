'''This module implements concrete agent controllers for the rollout worker'''
import bisect
import math
import numpy as np
import os
import random
import rospkg
import rospy
import threading

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SpawnModel
from geometry_msgs.msg import Pose
from rosgraph_msgs.msg import Clock

from markov.agent_ctrl.constants import ConfigParams, BOT_CAR_Z
from markov.track_geom.constants import SET_MODEL_STATE, SPAWN_SDF_MODEL, ObstacleDimensions
from markov.track_geom.track_data import TrackData, TrackLine
from markov.track_geom.utils import euler_to_quaternion
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.rospy_wrappers import ServiceProxyWrapper
from markov import utils
from markov.reset.constants import AgentPhase, AgentInfo
from markov.deepracer_exceptions import GenericRolloutException

from scipy.interpolate import splprep, spalde
from shapely.geometry import Point
from shapely.geometry.polygon import LineString

SPLINE_DEGREE = 3

class BotCarsCtrl(AgentCtrlInterface):
    def __init__(self):
        self.track_data = TrackData.get_instance()
        self.lock = threading.Lock()

        # Read ros parameters
        self.num_bot_cars = int(rospy.get_param("NUMBER_OF_BOT_CARS", 0))
        self.min_bot_car_dist = float(rospy.get_param("MIN_DISTANCE_BETWEEN_BOT_CARS", 2.0))
        self.randomize = utils.str2bool(rospy.get_param("RANDOMIZE_BOT_CAR_LOCATIONS", False))
        self.bot_car_speed = float(rospy.get_param("BOT_CAR_SPEED", 0.2))
        self.is_lane_change = utils.str2bool(rospy.get_param("IS_LANE_CHANGE", False))
        self.lower_lane_change_time = float(rospy.get_param("LOWER_LANE_CHANGE_TIME", 3.0))
        self.upper_lane_change_time = float(rospy.get_param("UPPER_LANE_CHANGE_TIME", 5.0))
        self.lane_change_distance = float(rospy.get_param("LANE_CHANGE_DISTANCE", 1.0))
        self.penalty_seconds = float(rospy.get_param("PENALTY_SECONDS", 2.0))
        self.lane_change_duration = self.lane_change_distance/self.bot_car_speed
        self.bot_car_names = ["bot_car_{}".format(i) for i in range(self.num_bot_cars)]
        self.bot_car_poses = []
        self.bot_car_progresses = {}
        self.bot_car_phase = AgentPhase.RUN.value
        self.bot_car_dimensions = ObstacleDimensions.BOT_CAR_DIMENSION
        self.bot_car_crash_count = 0
        self.pause_end_time = 0.0

        # Wait for ros services
        rospy.wait_for_service(SET_MODEL_STATE)
        rospy.wait_for_service(SPAWN_SDF_MODEL)
        self.set_model_state = ServiceProxyWrapper(SET_MODEL_STATE, SetModelState)
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)

        # Build splines for inner/outer lanes
        self.inner_lane = self._build_lane(self.track_data._inner_lane_)
        self.outer_lane = self._build_lane(self.track_data._outer_lane_)

        # Spawn the bot cars
        self._reset_sim_time()
        self._spawn_bot_cars()

        # Subscribe to the Gazebo clock and model states
        rospy.Subscriber('/clock', Clock, self._update_sim_time)

    def _build_lane(self, lane):
        '''Take in the track lane and return a track lane dictionary

        Args:
            lane (TrackLine): TrackLine instance

        Returns:
            dict: Dictionary contains input track lane, track lane point distance,
                  prepared track lane spline.
        '''
        center_line = self.track_data._center_line_
        lane_dists = [center_line.project(Point(c)) for c in lane.coords]
        # projecting inner/outer lane into center line cannot
        # guarantee monotonic increase along starting and ending position
        # if wrap around along start (more than half of track length),
        # subtract track length
        for i in range(len(lane_dists)):
            if lane_dists[i] < 0.5 * center_line.length:
                break
            lane_dists[i] -= center_line.length
        # if wrap around along finish (less than half of track length),
        # add track length
        for i in range(len(lane_dists) - 1, 0, -1):
            if lane_dists[i] > 0.5 * center_line.length:
                break
            lane_dists[i] += center_line.length
        u,ui = np.unique(lane_dists, return_index=True)
        x = np.array(lane.coords.xy)[:,ui]
        if u[0] > 0.0:
            p0 = lane.interpolate(lane.project(Point(center_line.coords[0])))
            u[0] = 0.0
            x[:,:1] = p0.xy
        if u[-1] < center_line.length:
            pN = lane.interpolate(lane.project(Point(center_line.coords[-1])))
            u[-1] = center_line.length
            x[:,-1:] = pN.xy
        if self.track_data.is_loop:
            x[:,-1] = x[:,0]
            lane_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0, per=1)
        else:
            lane_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0)
        return {"track_line": lane,
                "dists": lane_dists,
                "spline": lane_spline}

    def _reset_sim_time(self):
        '''reset simulation start time
        '''
        sim_time = rospy.get_rostime()
        self.start_sim_time = self.current_sim_time = sim_time.secs + 1.e-9*sim_time.nsecs

    def _update_sim_time(self, sim_time):
        '''Gazebo clock call back to update current time

        Args:
            sim_time (Clock): gazebo clock
        '''
        self.current_sim_time = sim_time.clock.secs + 1.e-9*sim_time.clock.nsecs

    def _get_dist_from_sim_time(self, initial_dist, sim_time):
        '''Get the bot car travel distance since the simulation start time

        Args:
            initial_dist (float): bot car initial distance
            sim_time (float): current simulation time

        Returns:
            float: current bot car distance
        '''
        seconds_elapsed = sim_time - self.start_sim_time - self.bot_car_crash_count * \
                          self.penalty_seconds
        bot_car_traveled_dist = seconds_elapsed * self.bot_car_speed
        bot_car_center_dist = (initial_dist + bot_car_traveled_dist) \
                              % self.track_data._center_line_.length
        return bot_car_center_dist

    def _eval_spline(self, initial_dist, sim_time, spline):
        '''Use spline to generate point

        Args:
            initial_dist (float): bot car initial distance
            sim_time (float): current simulation time
            spline (splprep): B-spline representation of an N-dimensional curve.
            https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html

        Returns:
            spalde: Evaluate all derivatives of a B-spline.
            https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.interpolate.spalde.html
        '''
        center_line = self.track_data._center_line_
        dist = self._get_dist_from_sim_time(initial_dist, sim_time)
        min_dist = spline[0][SPLINE_DEGREE]
        max_dist = spline[0][-SPLINE_DEGREE-1]
        if dist < min_dist: dist += center_line.length
        if dist > max_dist: dist -= center_line.length
        return spalde(max(min_dist, min(dist, max_dist)), spline)

    def _compute_bot_car_initial_states(self):
        '''Compute the bot car initial distance and spline
        '''
        # Start with equally spaced
        bot_car_start_dist = self.min_bot_car_dist
        bot_car_end_dist = self.track_data._center_line_.length - 1.0
        bot_cars_initial_dists = np.linspace(bot_car_start_dist, bot_car_end_dist,
                                             self.num_bot_cars)

        # Perturb to achieve randomness
        self.bot_cars_lanes = []
        self.bot_cars_opposite_lanes = []
        self.bot_cars_lane_change_end_times = [0.0] * self.num_bot_cars if self.is_lane_change \
                                              else [float('inf')] * self.num_bot_cars
        lane_choices = (self.inner_lane, self.outer_lane)
        if self.randomize:
            i_bot_car = list(range(self.num_bot_cars))
            random.shuffle(i_bot_car)
            for i in i_bot_car:
                lo = bot_car_start_dist if (i == 0) \
                     else bot_cars_initial_dists[i-1] + self.min_bot_car_dist
                hi = bot_car_end_dist if (i == self.num_bot_cars-1) \
                     else bot_cars_initial_dists[i+1] - self.min_bot_car_dist
                if lo < hi:
                    bot_cars_initial_dists[i] = random.uniform(lo, hi)

            # Select a random lane for each bot car
            for _ in bot_cars_initial_dists:
                use_outer_lane = random.choice((False, True))
                self.bot_cars_lanes.append(lane_choices[use_outer_lane])
                self.bot_cars_opposite_lanes.append(lane_choices[not use_outer_lane])
        else:
            # Alternate between lanes for each bot car
            use_outer_lane = False
            for _ in bot_cars_initial_dists:
                self.bot_cars_lanes.append(lane_choices[use_outer_lane])
                self.bot_cars_opposite_lanes.append(lane_choices[not use_outer_lane])
                use_outer_lane = not use_outer_lane

        # Minimal critical section
        with self.lock:
            self.bot_cars_initial_dists = bot_cars_initial_dists
            self.bot_car_splines = [lane["spline"] for lane in self.bot_cars_lanes]

    def _compute_bot_car_lane_changes(self):
        '''Compute the bot car lane change splines and update bot car lane change end times
        '''
        center_line = self.track_data._center_line_

        bot_car_splines = []
        for i_bot_car, lane_change_end_time in enumerate(self.bot_cars_lane_change_end_times):

            # See if the last lane change has finished
            if self.current_sim_time >= lane_change_end_time:
                # Swap lanes
                if lane_change_end_time > 0.0:
                    self.bot_cars_lanes[i_bot_car], self.bot_cars_opposite_lanes[i_bot_car] = \
                        self.bot_cars_opposite_lanes[i_bot_car], self.bot_cars_lanes[i_bot_car]

                start_lane = self.bot_cars_lanes[i_bot_car]
                start_lane_line = start_lane["track_line"]
                start_lane_dists = start_lane["dists"]
                start_lane_spline = start_lane["spline"]
                end_lane = self.bot_cars_opposite_lanes[i_bot_car]
                end_lane_line = end_lane["track_line"]
                end_lane_dists = end_lane["dists"]
                end_lane_spline = end_lane["spline"]

                # Set the next lane change time
                lane_change_end_time = self.bot_cars_lane_change_end_times[i_bot_car] = \
                    self.current_sim_time \
                    + random.uniform(self.lower_lane_change_time, self.upper_lane_change_time) \
                    + self.lane_change_duration
                lane_change_start_time = lane_change_end_time - self.lane_change_duration

                # Get center dists for relevant lane change times
                initial_dist = self.bot_cars_initial_dists[i_bot_car]
                current_dist = self._get_dist_from_sim_time(initial_dist, self.current_sim_time)
                start_dist = self._get_dist_from_sim_time(initial_dist, lane_change_start_time)
                end_dist = self._get_dist_from_sim_time(initial_dist, lane_change_end_time)
                end_offset = 0.0 if (start_dist < end_dist) else center_line.length

                # Grab start/end lane points from the times
                start_lane_point = Point(np.array(self._eval_spline(initial_dist,
                                                                    lane_change_start_time,
                                                                    start_lane_spline))[:,0])
                end_lane_point = Point(np.array(self._eval_spline(initial_dist,
                                                                  lane_change_end_time,
                                                                  end_lane_spline))[:,0])

                # Find prev/next points on each lane
                current_prev_index = bisect.bisect_left(start_lane_dists, current_dist) - 1
                start_prev_index = bisect.bisect_left(start_lane_dists, start_dist) - 1
                end_next_index = bisect.bisect_right(end_lane_dists, end_dist)

                # Define intervals on start/end lanes to build the spline from
                num_start_coords = len(start_lane_line.coords)
                num_end_coords = len(end_lane_line.coords)
                if self.track_data.is_loop:
                    num_start_coords -= 1
                    num_end_coords -= 1
                start_index_0 = (current_prev_index - 3) % num_start_coords
                start_index_1 = start_prev_index % num_start_coords
                end_index_0 = end_next_index % num_end_coords
                end_index_1 = (end_next_index + 3) % num_end_coords

                # Grab waypoint indices for these intervals (some corner cases here...)
                if start_index_0 < start_index_1:
                    start_indices = list(range(start_index_0, start_index_1 + 1))
                    start_offsets = [0.0] * len(start_indices)
                else:
                    start_indices_0 = list(range(start_index_0, num_start_coords))
                    start_indices_1 = list(range(start_index_1 + 1))
                    start_indices = start_indices_0 + start_indices_1
                    start_offsets = [-center_line.length] * len(start_indices_0) \
                                    + [0.0] * len(start_indices_1)
                if end_index_0 < end_index_1:
                    end_indices = list(range(end_index_0, end_index_1 + 1))
                    end_offsets = [end_offset] * len(end_indices)
                else:
                    end_indices_0 = list(range(end_index_0, num_end_coords))
                    end_indices_1 = list(range(end_index_1 + 1))
                    end_indices = end_indices_0 + end_indices_1
                    end_offsets = [end_offset] * len(end_indices_0) \
                                  + [end_offset + center_line.length] * len(end_indices_1)

                # Build the spline
                u = np.hstack((
                    np.array(start_lane_dists)[start_indices] + np.array(start_offsets),
                    start_dist,
                    end_dist + end_offset,
                    np.array(end_lane_dists)[end_indices] + np.array(end_offsets)))
                x = np.hstack((
                    np.array(start_lane_line.coords.xy)[:,start_indices],
                    start_lane_point.xy,
                    end_lane_point.xy,
                    np.array(end_lane_line.coords.xy)[:,end_indices]))
                u,ui = np.unique(u, return_index=True)
                x = x[:,ui]
                bot_car_spline, _ = splprep(x, u=u, k=SPLINE_DEGREE, s=0)
                bot_car_splines.append((i_bot_car, bot_car_spline))

        # Minimal critical section
        if bot_car_splines:
            with self.lock:
                for i_bot_car, bot_car_spline in bot_car_splines:
                    self.bot_car_splines[i_bot_car] = bot_car_spline

    def _compute_bot_car_poses(self):
        '''Compute bot car poses

        Returns:
            list: list of bot car Pose instance
        '''
        # Evaluate the splines
        with self.lock:
            bot_cars_spline_derivs = \
                [self._eval_spline(initial_dist, self.current_sim_time, spline)
                 for initial_dist, spline in zip(self.bot_cars_initial_dists,
                                                 self.bot_car_splines)]

        # Compute the bot car poses
        bot_car_poses = []
        for bot_car_spline_derivs in bot_cars_spline_derivs:
            bot_car_x, bot_car_y = bot_car_spline_derivs[0][0], bot_car_spline_derivs[1][0]
            bot_car_dx, bot_car_dy = bot_car_spline_derivs[0][1], bot_car_spline_derivs[1][1]
            bot_car_yaw = math.atan2(bot_car_dy, bot_car_dx)
            bot_car_orientation = euler_to_quaternion(yaw=bot_car_yaw, pitch=0, roll=0)

            bot_car_pose = Pose()
            bot_car_pose.position.x = bot_car_x
            bot_car_pose.position.y = bot_car_y
            bot_car_pose.position.z = BOT_CAR_Z
            bot_car_pose.orientation.x = bot_car_orientation[0]
            bot_car_pose.orientation.y = bot_car_orientation[1]
            bot_car_pose.orientation.z = bot_car_orientation[2]
            bot_car_pose.orientation.w = bot_car_orientation[3]
            bot_car_poses.append(bot_car_pose)

        return bot_car_poses

    def _spawn_bot_cars(self):
        '''Spawn the bot cars and initialize track data bot car objects
        '''
        self._compute_bot_car_initial_states()
        self.bot_car_poses = self._compute_bot_car_poses()

        rospack = rospkg.RosPack()
        deepracer_path = rospack.get_path("deepracer_simulation_environment")
        bot_car_sdf_path = os.path.join(deepracer_path, "models", "bot_car", "model.sdf")
        with open(bot_car_sdf_path, "r") as fp:
            bot_car_sdf = fp.read()

        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            self.spawn_sdf_model(bot_car_name, bot_car_sdf, '/{}'.format(bot_car_name),
                                 bot_car_pose, '')
            self.track_data.initialize_object(bot_car_name, bot_car_pose, self.bot_car_dimensions)

    def _update_bot_cars(self):
        '''Update bot car objects locations
        '''
        self.bot_car_poses = self._compute_bot_car_poses()
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            bot_car_state = ModelState()
            bot_car_state.model_name = bot_car_name
            bot_car_state.pose = bot_car_pose
            bot_car_state.twist.linear.x = 0
            bot_car_state.twist.linear.y = 0
            bot_car_state.twist.linear.z = 0
            bot_car_state.twist.angular.x = 0
            bot_car_state.twist.angular.y = 0
            bot_car_state.twist.angular.z = 0
            self.set_model_state(bot_car_state)
            self.track_data.reset_object(bot_car_name, bot_car_pose)

    @property
    def action_space(self):
        return None

    def reset_agent(self):
        '''reset bot car when a new episode start by resetting simulation time and
        initial position.
        '''
        self.bot_car_crash_count = 0
        self._reset_sim_time()
        self._compute_bot_car_initial_states()
        self._update_bot_cars()

    def send_action(self, action):
        '''Send bot car action to Gazebo for rendering

        Args:
            action (int): index of action
        '''
        if self.bot_car_phase == AgentPhase.RUN.value:
            self._compute_bot_car_lane_changes()
            self._update_bot_cars()

    def update_agent(self, action):
        '''Update bot car status after action is taken

        Args:
            action (int): index of action

        Returns:
            dict: dictionary of bot car info after action is taken
        '''
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            bot_car_progress = self.track_data.get_norm_dist(
                Point(bot_car_pose.position.x, bot_car_pose.position.y)) * 100.0
            self.bot_car_progresses.update(
                {bot_car_name:{AgentInfo.CURRENT_PROGRESS.value:bot_car_progress}})
        return self.bot_car_progresses

    def judge_action(self, agents_info_map):
        '''Judge action to see whether reset is needed

        Args:
            agents_info_map: Dictionary contains all agents info with agent name as the key
                             and info as the value

        Returns:
            tuple: None, None, None

        Raises:
            GenericRolloutException: bot car pahse is not defined
        '''
        if self.bot_car_phase == AgentPhase.RUN.value:
            self.pause_end_time = self.current_sim_time
            for agent_name, _ in agents_info_map.items():
                # check racecar crash with a bot_car
                crashed_object_name = agents_info_map[agent_name]\
                    [AgentInfo.CRASHED_OBJECT_NAME.value] \
                    if AgentInfo.CRASHED_OBJECT_NAME.value in agents_info_map[agent_name] else None
                if 'racecar' in agent_name and \
                        crashed_object_name and 'bot_car' in crashed_object_name:
                    racecar_progress = agents_info_map[agent_name]\
                                                      [AgentInfo.CURRENT_PROGRESS.value]
                    bot_car_progress = agents_info_map[crashed_object_name]\
                                                      [AgentInfo.CURRENT_PROGRESS.value]
                    # transition to AgentPhase.PAUSE.value
                    if racecar_progress > bot_car_progress:
                        self.bot_cars_lane_change_end_times = \
                            [t + self.penalty_seconds for t in self.bot_cars_lane_change_end_times]
                        self.bot_car_crash_count += 1
                        self.pause_end_time += self.penalty_seconds
                        self.bot_car_phase = AgentPhase.PAUSE.value
                        break
        elif self.bot_car_phase == AgentPhase.PAUSE.value:
            # transition to AgentPhase.RUN.value
            if self.current_sim_time > self.pause_end_time:
                self.bot_car_phase = AgentPhase.RUN.value
        else:
            raise GenericRolloutException('bot car phase {} is not defined'.\
                  format(self.bot_car_phase))
        return None, None, None

    def finish_episode(self):
        pass
