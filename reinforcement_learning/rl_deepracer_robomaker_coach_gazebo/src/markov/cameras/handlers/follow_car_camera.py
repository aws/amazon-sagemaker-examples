import math

import numpy as np
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose
from markov.cameras import utils
from markov.cameras.abs_camera import AbstractCamera
from markov.gazebo_tracker.trackers.set_model_state_tracker import SetModelStateTracker
from markov.track_geom.utils import apply_orientation, euler_to_quaternion, quaternion_to_euler


class FollowCarCamera(AbstractCamera):
    """this module is for follow-car camera"""

    name = "follow_car_camera"

    def __init__(self, namespace=None, model_name=None):
        super(FollowCarCamera, self).__init__(
            FollowCarCamera.name, namespace=namespace, model_name=model_name
        )
        # Camera configuration constants
        self.look_down_angle_rad = math.pi / 6.0  # 30 degree
        self.cam_dist_offset = 1.2
        self.cam_fixed_height = 1.0
        self.damping = 1.0
        # Camera states
        self.last_yaw = 0.0
        self.last_camera_state = None

    def _get_sdf_string(self, camera_sdf_path):
        with open(camera_sdf_path, "r") as sdf_file:
            camera_sdf = sdf_file.read()
        return camera_sdf

    def _get_initial_camera_pose(self, car_pose):
        _, _, car_yaw = quaternion_to_euler(
            x=car_pose.orientation.x,
            y=car_pose.orientation.y,
            z=car_pose.orientation.z,
            w=car_pose.orientation.w,
        )
        target_cam_quaternion = np.array(
            euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=car_yaw)
        )
        target_camera_location = np.array(
            [car_pose.position.x, car_pose.position.y, 0.0]
        ) + apply_orientation(target_cam_quaternion, np.array([-self.cam_dist_offset, 0, 0]))
        camera_model_pose = Pose()
        camera_model_pose.position.x = target_camera_location[0]
        camera_model_pose.position.y = target_camera_location[1]
        camera_model_pose.position.z = self.cam_fixed_height
        camera_model_pose.orientation.x = target_cam_quaternion[0]
        camera_model_pose.orientation.y = target_cam_quaternion[1]
        camera_model_pose.orientation.z = target_cam_quaternion[2]
        camera_model_pose.orientation.w = target_cam_quaternion[3]
        self.last_yaw = car_yaw

        return camera_model_pose

    def _reset(self, car_pose):
        camera_model_pose = self._get_initial_camera_pose(car_pose)
        camera_model_state = ModelState()
        camera_model_state.model_name = self.model_name
        camera_model_state.pose = camera_model_pose
        self.last_camera_state = camera_model_state
        SetModelStateTracker.get_instance().set_model_state(camera_model_state)

    def _update(self, car_pose, delta_time):
        # Calculate target Camera position based on car position
        _, _, car_yaw = quaternion_to_euler(
            x=car_pose.orientation.x,
            y=car_pose.orientation.y,
            z=car_pose.orientation.z,
            w=car_pose.orientation.w,
        )
        # Linear Interpolate Yaw angle
        car_yaw = utils.lerp_angle_rad(self.last_yaw, car_yaw, delta_time * self.damping)
        target_cam_quaternion = np.array(
            euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=car_yaw)
        )
        target_camera_location = np.array(
            [car_pose.position.x, car_pose.position.y, 0.0]
        ) + apply_orientation(target_cam_quaternion, np.array([-self.cam_dist_offset, 0, 0]))
        target_camera_location_2d = target_camera_location[0:2]

        # Linear interpolate Camera position to target position
        cur_camera_2d_pos = np.array(
            [self.last_camera_state.pose.position.x, self.last_camera_state.pose.position.y]
        )
        new_cam_pos_2d = utils.lerp(
            cur_camera_2d_pos, target_camera_location_2d, delta_time * self.damping
        )

        # Calculate camera rotation quaternion based on lookAt yaw
        look_at_yaw = utils.get_angle_between_two_points_2d_rad(
            self.last_camera_state.pose.position, car_pose.position
        )
        cam_quaternion = euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=look_at_yaw)

        # Configure Camera Model State
        camera_model_state = ModelState()
        camera_model_state.model_name = self.model_name
        camera_model_state.pose.position.x = new_cam_pos_2d[0]
        camera_model_state.pose.position.y = new_cam_pos_2d[1]
        camera_model_state.pose.position.z = self.cam_fixed_height
        camera_model_state.pose.orientation.x = cam_quaternion[0]
        camera_model_state.pose.orientation.y = cam_quaternion[1]
        camera_model_state.pose.orientation.z = cam_quaternion[2]
        camera_model_state.pose.orientation.w = cam_quaternion[3]
        SetModelStateTracker.get_instance().set_model_state(camera_model_state)

        self.last_camera_state = camera_model_state
        self.last_yaw = car_yaw
