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

from markov.track_geom.constants import SET_MODEL_STATE, SPAWN_URDF_MODEL, ObstacleDimensions
from markov.track_geom.track_data import TrackData, TrackLine
from markov.track_geom.utils import euler_to_quaternion
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.rospy_wrappers import ServiceProxyWrapper
from markov import utils

from scipy.interpolate import splprep, spalde
from shapely.geometry import Point
from shapely.geometry.polygon import LineString

SPLINE_DEGREE = 2

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
        self.lane_change_duration = self.lane_change_distance/self.bot_car_speed
        self.bot_car_names = ["bot_car_{}".format(i) for i in range(self.num_bot_cars)]
        self.bot_car_dimensions = ObstacleDimensions.BOT_CAR_DIMENSION

        # Wait for ros services
        rospy.wait_for_service(SET_MODEL_STATE)
        rospy.wait_for_service(SPAWN_URDF_MODEL)
        self.set_model_state = ServiceProxyWrapper(SET_MODEL_STATE, SetModelState)
        self.spawn_urdf_model = ServiceProxyWrapper(SPAWN_URDF_MODEL, SpawnModel)
        self.bot_car_urdf = rospy.get_param('robot_description_bot')

        # Build splines for inner/outer lanes
        self.inner_lane = self._build_lane(self.track_data._inner_lane_)
        self.outer_lane = self._build_lane(self.track_data._outer_lane_)

        # Spawn the bot cars
        self._reset_sim_time()
        self._spawn_bot_cars()

        # Subscribe to the Gazebo clock and model states
        rospy.Subscriber('/clock', Clock, self._update_sim_time)

    def _build_lane(self, lane):
        center_line = self.track_data._center_line_
        lane_dists = [center_line.project(Point(c)) for c in lane.coords]
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
        sim_time = rospy.get_rostime()
        self.start_sim_time = self.current_sim_time = sim_time.secs + 1.e-9*sim_time.nsecs

    def _update_sim_time(self, sim_time):
        self.current_sim_time = sim_time.clock.secs + 1.e-9*sim_time.clock.nsecs

    def _get_dist_from_sim_time(self, initial_dist, sim_time):
        seconds_elapsed = sim_time - self.start_sim_time
        bot_car_traveled_dist = seconds_elapsed * self.bot_car_speed
        bot_car_center_dist = (initial_dist + bot_car_traveled_dist) \
                              % self.track_data._center_line_.length
        return bot_car_center_dist

    def _eval_spline(self, initial_dist, sim_time, spline):
        center_line = self.track_data._center_line_
        dist = self._get_dist_from_sim_time(initial_dist, sim_time)
        min_dist = spline[0][SPLINE_DEGREE]
        max_dist = spline[0][-SPLINE_DEGREE-1]
        if dist < min_dist: dist += center_line.length
        if dist > max_dist: dist -= center_line.length
        return spalde(dist, spline)

    def _compute_bot_car_initial_states(self):

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
                start_index_1 = start_prev_index
                end_index_0 = end_next_index
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
            bot_car_pose.position.z = 0.0
            bot_car_pose.orientation.x = bot_car_orientation[0]
            bot_car_pose.orientation.y = bot_car_orientation[1]
            bot_car_pose.orientation.z = bot_car_orientation[2]
            bot_car_pose.orientation.w = bot_car_orientation[3]
            bot_car_poses.append(bot_car_pose)

        return bot_car_poses

    def _spawn_bot_cars(self):
        self._compute_bot_car_initial_states()
        bot_car_poses = self._compute_bot_car_poses()
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, bot_car_poses):
            self.spawn_urdf_model(bot_car_name, self.bot_car_urdf, '/{}'.format(bot_car_name),
                                  bot_car_pose, '')
            self.track_data.initialize_object(bot_car_name, bot_car_pose, self.bot_car_dimensions)

    def _update_bot_cars(self):
        bot_car_poses = self._compute_bot_car_poses()
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, bot_car_poses):
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
        self._reset_sim_time()
        self._compute_bot_car_initial_states()
        self._update_bot_cars()

    def send_action(self, action):
        self._compute_bot_car_lane_changes()
        self._update_bot_cars()

    def judge_action(self, action):
        return None, None, None

    def finish_episode(self):
        pass

    def clear_data(self):
        pass
