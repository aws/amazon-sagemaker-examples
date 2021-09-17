"""This module implements concrete agent controllers for the rollout worker"""
import math
import os
import random
import threading

import numpy as np
import rospkg
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
from markov import utils
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.agent_ctrl.constants import BOT_CAR_Z
from markov.agent_ctrl.utils import get_normalized_progress
from markov.domain_randomizations.constants import ModelRandomizerType
from markov.domain_randomizations.randomizer_manager import RandomizerManager
from markov.domain_randomizations.visual.model_visual_randomizer import ModelVisualRandomizer
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.gazebo_tracker.constants import TrackerPriority
from markov.gazebo_tracker.trackers.set_model_state_tracker import SetModelStateTracker
from markov.log_handler.deepracer_exceptions import GenericRolloutException
from markov.reset.constants import AgentInfo, AgentPhase
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SPAWN_SDF_MODEL, ObstacleDimensions, TrackLane
from markov.track_geom.spline.lane_change_spline import LaneChangeSpline
from markov.track_geom.spline.track_spline import TrackSpline
from markov.track_geom.track_data import TrackData
from markov.track_geom.utils import euler_to_quaternion
from shapely.geometry import Point


class BotCarsCtrl(AgentCtrlInterface, AbstractTracker):
    def __init__(self):
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
        self.lane_change_duration = self.lane_change_distance / self.bot_car_speed
        self.bot_car_names = ["bot_car_{}".format(i) for i in range(self.num_bot_cars)]
        self.bot_car_poses = []
        self.bot_car_progresses = {}
        self.bot_car_phase = AgentPhase.RUN.value
        self.bot_car_dimensions = ObstacleDimensions.BOT_CAR_DIMENSION
        self.bot_car_crash_count = 0
        self.pause_duration = 0.0

        # track date
        self.track_data = TrackData.get_instance()
        self.reverse_dir = self.track_data.reverse_dir

        # Wait for ros services
        rospy.wait_for_service(SPAWN_SDF_MODEL)
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)

        # Build splines for inner/outer lanes
        self.inner_lane = TrackSpline(lane_name=TrackLane.INNER_LANE.value)
        self.outer_lane = TrackSpline(lane_name=TrackLane.OUTER_LANE.value)

        # Spawn the bot cars
        self.current_sim_time = 0.0
        self._reset_sim_time()
        self._spawn_bot_cars()

        self._configure_randomizer()
        AbstractTracker.__init__(self, priority=TrackerPriority.HIGH)

    def _configure_randomizer(self):
        """configure domain randomizer"""
        for bot_car_name in self.bot_car_names:
            RandomizerManager.get_instance().add(
                ModelVisualRandomizer(
                    model_name=bot_car_name, model_randomizer_type=ModelRandomizerType.VISUAL
                )
            )

    def _reset_sim_time(self):
        """reset simulation start time"""
        sim_time = rospy.get_rostime()
        self.start_sim_time = self.current_sim_time = sim_time.secs + 1.0e-9 * sim_time.nsecs

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        self.current_sim_time = sim_time.clock.secs + 1.0e-9 * sim_time.clock.nsecs
        if self.pause_duration > 0.0:
            self.pause_duration -= delta_time

    def _get_dist_from_sim_time(self, initial_dist, sim_time):
        """Get the bot car travel distance since the simulation start time

        Args:
            initial_dist (float): bot car initial distance
            sim_time (float): current simulation time

        Returns:
            float: current bot car distance
        """
        seconds_elapsed = (
            sim_time - self.start_sim_time - self.bot_car_crash_count * self.penalty_seconds
        )
        bot_car_traveled_dist = seconds_elapsed * self.bot_car_speed
        bot_car_center_dist = (
            initial_dist + bot_car_traveled_dist
        ) % self.track_data.center_line.length
        return bot_car_center_dist

    def _compute_bot_car_initial_states(self):
        """Compute the bot car initial distance and spline"""
        # Start with equally spaced
        bot_car_start_dist = self.min_bot_car_dist
        bot_car_end_dist = self.track_data.center_line.length - 1.0
        bot_cars_initial_dists = np.linspace(
            bot_car_start_dist, bot_car_end_dist, self.num_bot_cars
        )

        # Perturb to achieve randomness
        self.bot_cars_lane_splines = []
        self.bot_cars_opposite_lane_splines = []
        self.bot_cars_lane_change_end_times = (
            [0.0] * self.num_bot_cars if self.is_lane_change else [float("inf")] * self.num_bot_cars
        )
        lane_choices = (self.inner_lane, self.outer_lane)
        if self.randomize:
            i_bot_car = list(range(self.num_bot_cars))
            random.shuffle(i_bot_car)
            for i in i_bot_car:
                lo = (
                    bot_car_start_dist
                    if (i == 0)
                    else bot_cars_initial_dists[i - 1] + self.min_bot_car_dist
                )
                hi = (
                    bot_car_end_dist
                    if (i == self.num_bot_cars - 1)
                    else bot_cars_initial_dists[i + 1] - self.min_bot_car_dist
                )
                if lo < hi:
                    bot_cars_initial_dists[i] = random.uniform(lo, hi)

            # Select a random lane for each bot car
            for _ in bot_cars_initial_dists:
                use_outer_lane = random.choice((False, True))
                self.bot_cars_lane_splines.append(lane_choices[use_outer_lane])
                self.bot_cars_opposite_lane_splines.append(lane_choices[not use_outer_lane])
        else:
            # Alternate between lanes for each bot car
            use_outer_lane = False
            for _ in bot_cars_initial_dists:
                self.bot_cars_lane_splines.append(lane_choices[use_outer_lane])
                self.bot_cars_opposite_lane_splines.append(lane_choices[not use_outer_lane])
                use_outer_lane = not use_outer_lane

        # Minimal critical section
        with self.lock:
            self.bot_cars_initial_dists = bot_cars_initial_dists
            self.bot_cars_trajectories = list(self.bot_cars_lane_splines)

    def _compute_bot_car_lane_changes(self):
        """Compute the bot car lane change splines and update bot car lane change end times"""
        bot_cars_trajectories = []
        for i_bot_car, lane_change_end_time in enumerate(self.bot_cars_lane_change_end_times):

            # See if the last lane change has finished
            if self.current_sim_time >= lane_change_end_time:
                # Swap lanes
                if lane_change_end_time > 0.0:
                    (
                        self.bot_cars_lane_splines[i_bot_car],
                        self.bot_cars_opposite_lane_splines[i_bot_car],
                    ) = (
                        self.bot_cars_opposite_lane_splines[i_bot_car],
                        self.bot_cars_lane_splines[i_bot_car],
                    )
                # Get start lane and end lane
                start_lane = self.bot_cars_lane_splines[i_bot_car]
                end_lane = self.bot_cars_opposite_lane_splines[i_bot_car]
                # Get initial and curent distance
                initial_dist = self.bot_cars_initial_dists[i_bot_car]
                current_dist = self._get_dist_from_sim_time(initial_dist, self.current_sim_time)
                # Set the next lane change time
                lane_change_end_time = self.bot_cars_lane_change_end_times[i_bot_car] = (
                    self.current_sim_time
                    + random.uniform(self.lower_lane_change_time, self.upper_lane_change_time)
                    + self.lane_change_duration
                )
                lane_change_start_time = lane_change_end_time - self.lane_change_duration
                # Set the next lane change distance
                lane_change_start_dist = self._get_dist_from_sim_time(
                    initial_dist, lane_change_start_time
                )
                lane_change_end_dist = self._get_dist_from_sim_time(
                    initial_dist, lane_change_end_time
                )
                # Get bot car lane change spline
                bot_car_lane_change_spline = LaneChangeSpline(
                    start_lane=start_lane,
                    end_lane=end_lane,
                    current_dist=current_dist,
                    lane_change_start_dist=lane_change_start_dist,
                    lane_change_end_dist=lane_change_end_dist,
                )

                bot_cars_trajectories.append((i_bot_car, bot_car_lane_change_spline))

        # Minimal critical section
        if bot_cars_trajectories:
            with self.lock:
                for i_bot_car, bot_car_lane_change_spline in bot_cars_trajectories:
                    self.bot_cars_trajectories[i_bot_car] = bot_car_lane_change_spline

    def _compute_bot_car_poses(self):
        """Compute bot car poses

        Returns:
            list: list of bot car Pose instance
        """
        # Evaluate the splines
        with self.lock:
            bot_cars_spline_derivs = [
                spline.eval_spline(
                    self._get_dist_from_sim_time(initial_dist, self.current_sim_time)
                )
                for initial_dist, spline in zip(
                    self.bot_cars_initial_dists, self.bot_cars_trajectories
                )
            ]

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
        """Spawn the bot cars and initialize track data bot car objects"""
        self._compute_bot_car_initial_states()
        self.bot_car_poses = self._compute_bot_car_poses()

        rospack = rospkg.RosPack()
        deepracer_path = rospack.get_path("deepracer_simulation_environment")
        bot_car_sdf_path = os.path.join(deepracer_path, "models", "bot_car", "model.sdf")
        with open(bot_car_sdf_path, "r") as fp:
            bot_car_sdf = fp.read()

        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            self.spawn_sdf_model(
                bot_car_name, bot_car_sdf, "/{}".format(bot_car_name), bot_car_pose, ""
            )
            self.track_data.initialize_object(bot_car_name, bot_car_pose, self.bot_car_dimensions)

    def _update_bot_cars(self):
        """Update bot car objects locations"""
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
            SetModelStateTracker.get_instance().set_model_state(bot_car_state)

    def _update_track_data_bot_car_poses(self):
        """update bot car poses in track data"""
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            self.track_data.update_object_pose(bot_car_name, bot_car_pose)

    @property
    def action_space(self):
        return None

    def reset_agent(self):
        """reset bot car when a new episode start by resetting simulation time,
        initial position and reverse direction potentially.
        """
        self.bot_car_crash_count = 0
        self.bot_car_phase = AgentPhase.RUN.value
        self.pause_duration = 0.0
        if self.reverse_dir != self.track_data.reverse_dir:
            self.reverse_dir = self.track_data.reverse_dir
            self.inner_lane.build_spline()
            self.outer_lane.build_spline()
        self._reset_sim_time()
        self._compute_bot_car_initial_states()
        self._update_bot_cars()
        self._update_track_data_bot_car_poses()

    def send_action(self, action):
        """Send bot car action to Gazebo for rendering

        Args:
            action (int): index of action
        """
        if self.bot_car_phase == AgentPhase.RUN.value:
            self._compute_bot_car_lane_changes()
            self._update_bot_cars()

    def update_agent(self, action):
        """Update bot car status after action is taken

        Args:
            action (int): index of action

        Returns:
            dict: dictionary of bot car info after action is taken
        """
        self._update_track_data_bot_car_poses()
        for bot_car_name, bot_car_pose in zip(self.bot_car_names, self.bot_car_poses):
            bot_car_progress = (
                self.track_data.get_norm_dist(
                    Point(bot_car_pose.position.x, bot_car_pose.position.y)
                )
                * 100.0
            )
            self.bot_car_progresses.update(
                {
                    bot_car_name: {
                        AgentInfo.CURRENT_PROGRESS.value: bot_car_progress,
                        AgentInfo.START_NDIST.value: 0.0,
                    }
                }
            )
        return self.bot_car_progresses

    def judge_action(self, agents_info_map):
        """Judge action to see whether reset is needed

        Args:
            agents_info_map: Dictionary contains all agents info with agent name as the key
                             and info as the value

        Returns:
            tuple: None, None, None

        Raises:
            GenericRolloutException: bot car phase is not defined
        """
        if self.bot_car_phase == AgentPhase.RUN.value:
            self.pause_duration = 0.0
            for agent_name, agent_info in agents_info_map.items():
                if not self.track_data.is_object_collidable(agent_name):
                    continue
                # check racecar crash with a bot_car
                crashed_object_name = (
                    agent_info[AgentInfo.CRASHED_OBJECT_NAME.value]
                    if AgentInfo.CRASHED_OBJECT_NAME.value in agent_info
                    else ""
                )
                # only trainable racecar agent has 'bot_car' as possible crashed object
                if "bot_car" in crashed_object_name:
                    racecar_progress = get_normalized_progress(
                        agent_info[AgentInfo.CURRENT_PROGRESS.value],
                        start_ndist=agent_info[AgentInfo.START_NDIST.value],
                    )
                    bot_car_info = agents_info_map[crashed_object_name]
                    bot_car_progress = get_normalized_progress(
                        bot_car_info[AgentInfo.CURRENT_PROGRESS.value],
                        start_ndist=bot_car_info[AgentInfo.START_NDIST.value],
                    )

                    # transition to AgentPhase.PAUSE.value
                    if racecar_progress > bot_car_progress:
                        self.bot_cars_lane_change_end_times = [
                            t + self.penalty_seconds for t in self.bot_cars_lane_change_end_times
                        ]
                        self.bot_car_crash_count += 1
                        self.pause_duration += self.penalty_seconds
                        self.bot_car_phase = AgentPhase.PAUSE.value
                        break
        elif self.bot_car_phase == AgentPhase.PAUSE.value:
            # transition to AgentPhase.RUN.value
            if self.pause_duration <= 0.0:
                self.bot_car_phase = AgentPhase.RUN.value
        else:
            raise GenericRolloutException(
                "bot car phase {} is not defined".format(self.bot_car_phase)
            )
        return None, None, None

    def finish_episode(self):
        pass
