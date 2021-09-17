"""This module implements concrete agent controllers for the rollout worker"""
import os
import random

import numpy as np
import rospkg
import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel
from markov import utils
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.agent_ctrl.constants import BOT_CAR_Z, OBSTACLE_NAME_PREFIX, OBSTACLE_Z
from markov.domain_randomizations.constants import ModelRandomizerType
from markov.domain_randomizations.randomizer_manager import RandomizerManager
from markov.domain_randomizations.visual.model_visual_randomizer import ModelVisualRandomizer
from markov.gazebo_tracker.trackers.set_model_state_tracker import SetModelStateTracker
from markov.reset.constants import AgentInfo
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.constants import SPAWN_SDF_MODEL, SPAWN_URDF_MODEL, ObstacleDimensions
from markov.track_geom.track_data import TrackData


class ObstaclesCtrl(AgentCtrlInterface):
    def __init__(self):
        # Read ros parameters
        # OBJECT_POSITIONS will overwrite NUMBER_OF_OBSTACLES and RANDOMIZE_OBSTACLE_LOCATIONS
        self.object_locations = rospy.get_param("OBJECT_POSITIONS", [])
        self.num_obstacles = (
            int(rospy.get_param("NUMBER_OF_OBSTACLES", 0))
            if not self.object_locations
            else len(self.object_locations)
        )
        self.min_obstacle_dist = float(rospy.get_param("MIN_DISTANCE_BETWEEN_OBSTACLES", 2.0))
        self.randomize = utils.str2bool(rospy.get_param("RANDOMIZE_OBSTACLE_LOCATIONS", False))
        self.use_bot_car = utils.str2bool(rospy.get_param("IS_OBSTACLE_BOT_CAR", False))
        self.obstacle_names = [
            "{}_{}".format(OBSTACLE_NAME_PREFIX, i) for i in range(self.num_obstacles)
        ]
        self.obstacle_dimensions = (
            ObstacleDimensions.BOT_CAR_DIMENSION
            if self.use_bot_car
            else ObstacleDimensions.BOX_OBSTACLE_DIMENSION
        )

        # track data
        self.track_data = TrackData.get_instance()

        # Wait for ros services
        rospy.wait_for_service(SPAWN_SDF_MODEL)
        rospy.wait_for_service(SPAWN_URDF_MODEL)
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)
        self.spawn_urdf_model = ServiceProxyWrapper(SPAWN_URDF_MODEL, SpawnModel)

        # Load the obstacle sdf/urdf
        obstacle_model_folder = "bot_car" if self.use_bot_car else "box_obstacle"
        rospack = rospkg.RosPack()
        deepracer_path = rospack.get_path("deepracer_simulation_environment")
        obstacle_sdf_path = os.path.join(
            deepracer_path, "models", obstacle_model_folder, "model.sdf"
        )
        with open(obstacle_sdf_path, "r") as fp:
            self.obstacle_sdf = fp.read()

        # Set obstacle poses and spawn the obstacles
        self.obstacle_poses = self._compute_obstacle_poses()
        self._spawn_obstacles()

        self._configure_randomizer()

    def _configure_randomizer(self):
        """configure domain randomizer"""
        for obstacle_names in self.obstacle_names:
            RandomizerManager.get_instance().add(
                ModelVisualRandomizer(
                    model_name=obstacle_names, model_randomizer_type=ModelRandomizerType.MODEL
                )
            )

    def _compute_obstacle_poses(self):
        obstacle_dists = []
        obstacle_lanes = []
        lane_choices = (self.track_data.inner_lane, self.track_data.outer_lane)
        # use fix obstacle locations
        if self.object_locations:
            for object_location in self.object_locations:
                # index 0 is obstacle_ndist and index 1 is obstacle_lane
                object_location = object_location.split(",")
                obstacle_dists.append(
                    float(object_location[0]) * self.track_data.center_line.length
                )
                # Inner lane is 1, outer lane is -1. If True, use outer lane
                obstacle_lanes.append(lane_choices[int(object_location[1]) == -1])
        else:
            # Start with equally spaced
            obstacle_start_dist = self.min_obstacle_dist
            obstacle_end_dist = self.track_data.center_line.length - 1.0
            obstacle_dists = np.linspace(obstacle_start_dist, obstacle_end_dist, self.num_obstacles)
            # Perturb to achieve randomness
            if self.randomize:
                i_obstacle = list(range(self.num_obstacles))
                random.shuffle(i_obstacle)
                for i in i_obstacle:
                    lo = (
                        obstacle_start_dist
                        if (i == 0)
                        else obstacle_dists[i - 1] + self.min_obstacle_dist
                    )
                    hi = (
                        obstacle_end_dist
                        if (i == self.num_obstacles - 1)
                        else obstacle_dists[i + 1] - self.min_obstacle_dist
                    )
                    if lo < hi:
                        obstacle_dists[i] = random.uniform(lo, hi)

                # Select a random lane for each obstacle
                for _ in obstacle_dists:
                    use_outer_lane = random.choice((False, True))
                    obstacle_lanes.append(lane_choices[use_outer_lane])
            else:
                # Alternate between lanes for each obstacle
                use_outer_lane = False
                for _ in obstacle_dists:
                    obstacle_lanes.append(lane_choices[use_outer_lane])
                    use_outer_lane = not use_outer_lane

        # Compute the obstacle poses
        obstacle_poses = []
        for obstacle_dist, obstacle_lane in zip(obstacle_dists, obstacle_lanes):
            obstacle_pose = obstacle_lane.interpolate_pose(
                obstacle_lane.project(self.track_data.center_line.interpolate(obstacle_dist))
            )
            if self.use_bot_car:
                obstacle_pose.position.z = BOT_CAR_Z
            else:
                obstacle_pose.position.z = OBSTACLE_Z
            obstacle_poses.append(obstacle_pose)

        # Return the poses
        return obstacle_poses

    def _spawn_obstacles(self):
        for obstacle_name, obstacle_pose in zip(self.obstacle_names, self.obstacle_poses):
            self.spawn_sdf_model(
                obstacle_name, self.obstacle_sdf, "/{}".format(obstacle_name), obstacle_pose, ""
            )
            self.track_data.initialize_object(
                obstacle_name, obstacle_pose, self.obstacle_dimensions
            )

    def _reset_obstacles(self):
        for obstacle_name, obstacle_pose in zip(self.obstacle_names, self.obstacle_poses):
            obstacle_state = ModelState()
            obstacle_state.model_name = obstacle_name
            obstacle_state.pose = obstacle_pose
            obstacle_state.twist.linear.x = 0
            obstacle_state.twist.linear.y = 0
            obstacle_state.twist.linear.z = 0
            obstacle_state.twist.angular.x = 0
            obstacle_state.twist.angular.y = 0
            obstacle_state.twist.angular.z = 0
            SetModelStateTracker.get_instance().set_model_state(obstacle_state)

    def _update_track_data_object_poses(self):
        """update object poses in track data"""
        for obstacle_name, obstacle_pose in zip(self.obstacle_names, self.obstacle_poses):
            self.track_data.update_object_pose(obstacle_name, obstacle_pose)

    @property
    def action_space(self):
        return None

    def reset_agent(self):
        self.obstacle_poses = self._compute_obstacle_poses()
        self._reset_obstacles()
        self._update_track_data_object_poses()

    def send_action(self, action):
        pass

    def update_agent(self, action):
        self._update_track_data_object_poses()
        return {}

    def judge_action(self, agents_info_map):
        for agent_name, agent_info in agents_info_map.items():
            # check racecar crash with a obstacle
            crashed_object_name = (
                agent_info[AgentInfo.CRASHED_OBJECT_NAME.value]
                if AgentInfo.CRASHED_OBJECT_NAME.value in agent_info
                else ""
            )
            # only trainable racecar agent has 'obstacle' as possible crashed object
            if OBSTACLE_NAME_PREFIX in crashed_object_name:
                self._reset_obstacles()
                break
        return None, None, None

    def finish_episode(self):
        pass
