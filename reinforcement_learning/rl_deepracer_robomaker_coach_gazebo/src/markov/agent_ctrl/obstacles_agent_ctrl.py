'''This module implements concrete agent controllers for the rollout worker'''
import numpy as np
import os
import random
import rospkg
import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SpawnModel
from markov.agent_ctrl.constants import ConfigParams, BOT_CAR_Z, OBSTACLE_Z
from markov.track_geom.constants import SET_MODEL_STATE, SPAWN_SDF_MODEL, SPAWN_URDF_MODEL, ObstacleDimensions
from markov.track_geom.track_data import TrackData
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface
from markov.rospy_wrappers import ServiceProxyWrapper
from markov import utils

class ObstaclesCtrl(AgentCtrlInterface):
    def __init__(self):
        self.track_data = TrackData.get_instance()
        # Read ros parameters
        # OBJECT_POSITIONS will overwrite NUMBER_OF_OBSTACLES and RANDOMIZE_OBSTACLE_LOCATIONS
        self.object_locations = rospy.get_param("OBJECT_POSITIONS", [])
        self.num_obstacles = int(rospy.get_param("NUMBER_OF_OBSTACLES", 0)) \
                             if not self.object_locations else len(self.object_locations)
        self.min_obstacle_dist = float(rospy.get_param("MIN_DISTANCE_BETWEEN_OBSTACLES", 2.0))
        self.randomize = utils.str2bool(rospy.get_param("RANDOMIZE_OBSTACLE_LOCATIONS", False))
        self.use_bot_car = utils.str2bool(rospy.get_param("IS_OBSTACLE_BOT_CAR", False))
        self.obstacle_names = ["obstacle_{}".format(i) for i in range(self.num_obstacles)]
        self.obstacle_dimensions = ObstacleDimensions.BOT_CAR_DIMENSION if self.use_bot_car \
                                   else ObstacleDimensions.BOX_OBSTACLE_DIMENSION

        # Wait for ros services
        rospy.wait_for_service(SET_MODEL_STATE)
        rospy.wait_for_service(SPAWN_SDF_MODEL)
        rospy.wait_for_service(SPAWN_URDF_MODEL)
        self.set_model_state = ServiceProxyWrapper(SET_MODEL_STATE, SetModelState)
        self.spawn_sdf_model = ServiceProxyWrapper(SPAWN_SDF_MODEL, SpawnModel)
        self.spawn_urdf_model = ServiceProxyWrapper(SPAWN_URDF_MODEL, SpawnModel)

        # Load the obstacle sdf/urdf
        obstacle_model_folder = "bot_car" if self.use_bot_car else "box_obstacle"
        rospack = rospkg.RosPack()
        deepracer_path = rospack.get_path("deepracer_simulation_environment")
        obstacle_sdf_path = os.path.join(deepracer_path, "models", obstacle_model_folder, "model.sdf")
        with open(obstacle_sdf_path, "r") as fp:
            self.obstacle_sdf = fp.read()

        # Spawn the obstacles
        self._spawn_obstacles()

    def _compute_obstacle_poses(self):
        obstacle_dists = []
        obstacle_lanes = []
        lane_choices = (self.track_data._inner_lane_, self.track_data._outer_lane_)
        # use fix obstacle locations
        if self.object_locations:
            for object_location in self.object_locations:
                # index 0 is obstacle_ndist and index 1 is obstacle_lane
                object_location = object_location.split(",")
                obstacle_dists.append(float(object_location[0]) * \
                                      self.track_data._center_line_.length)
                # Inner lane is 1, outer lane is -1. If True, use outer lane
                obstacle_lanes.append(lane_choices[int(object_location[1]) == -1])
        else:
            # Start with equally spaced
            obstacle_start_dist = self.min_obstacle_dist
            obstacle_end_dist = self.track_data._center_line_.length - 1.0
            obstacle_dists = np.linspace(obstacle_start_dist, obstacle_end_dist, self.num_obstacles)
            # Perturb to achieve randomness
            if self.randomize:
                i_obstacle = list(range(self.num_obstacles))
                random.shuffle(i_obstacle)
                for i in i_obstacle:
                    lo = obstacle_start_dist if (i == 0) \
                         else obstacle_dists[i-1] + self.min_obstacle_dist
                    hi = obstacle_end_dist if (i == self.num_obstacles-1) \
                         else obstacle_dists[i+1] - self.min_obstacle_dist
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
                obstacle_lane.project(self.track_data._center_line_.interpolate(obstacle_dist)))
            if self.use_bot_car:
                obstacle_pose.position.z = BOT_CAR_Z
            else:
                obstacle_pose.position.z = OBSTACLE_Z
            obstacle_poses.append(obstacle_pose)

        # Return the poses
        return obstacle_poses

    def _spawn_obstacles(self):
        obstacle_poses = self._compute_obstacle_poses()
        for obstacle_name, obstacle_pose in zip(self.obstacle_names, obstacle_poses):
            self.spawn_sdf_model(obstacle_name, self.obstacle_sdf, '/{}'.format(obstacle_name),
                                     obstacle_pose, '')
            self.track_data.initialize_object(obstacle_name, obstacle_pose, self.obstacle_dimensions)

    def _reset_obstacles(self):
        obstacle_poses = self._compute_obstacle_poses()
        for obstacle_name, obstacle_pose in zip(self.obstacle_names, obstacle_poses):
            obstacle_state = ModelState()
            obstacle_state.model_name = obstacle_name
            obstacle_state.pose = obstacle_pose
            obstacle_state.twist.linear.x = 0
            obstacle_state.twist.linear.y = 0
            obstacle_state.twist.linear.z = 0
            obstacle_state.twist.angular.x = 0
            obstacle_state.twist.angular.y = 0
            obstacle_state.twist.angular.z = 0
            self.set_model_state(obstacle_state)
            self.track_data.reset_object(obstacle_name, obstacle_pose)

    @property
    def action_space(self):
        return None

    def reset_agent(self):
        self._reset_obstacles()

    def send_action(self, action):
        pass

    def update_agent(self, action):
        return {}

    def judge_action(self, agents_info_map):
        return None, None, None

    def finish_episode(self):
        pass
