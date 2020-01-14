import numpy as np
import rospy
import math

from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from markov.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.track_data import TrackData
from markov.cameras import utils
from markov.track_geom.utils import euler_to_quaternion, apply_orientation
from markov.cameras.abs_camera import BaseCamera
from shapely.geometry import Point


class FollowTrackBehindCamera(BaseCamera):
    """this class is for follow-track-behind third-person camera singelton"""
    _instance_ = None
    name = "follow_track_behind_camera"

    @staticmethod
    def get_instance():
        """Method for geting a reference to the camera object"""
        if FollowTrackBehindCamera._instance_ is None:
            FollowTrackBehindCamera()
        return FollowTrackBehindCamera._instance_

    def __init__(self):
        if FollowTrackBehindCamera._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple follow track camera")
        super(FollowTrackBehindCamera, self).__init__(FollowTrackBehindCamera.name)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.model_state_client = ServiceProxyWrapper('/gazebo/set_model_state', SetModelState)
        self.track_data = TrackData.get_instance()

        # Camera configuration constants
        self.look_down_angle_rad = math.pi / 6.0  # 30 degree
        self.cam_dist_offset = 1.2
        self.cam_fixed_height = 1.0
        self.damping = 1.0

        # Camera states
        self.last_yaw = 0.0
        self.last_camera_state = None

        # there should be only one video camera instance
        FollowTrackBehindCamera._instance_ = self

    def _reset(self, car_model_state):
        camera_model_state = ModelState()
        camera_model_state.model_name = self.topic_name

        # Calculate target Camera position based on nearest center track from the car.
        # 1. Project the car position to 1-d global distance of track
        # 2. Minus track with offset and get the point of center track
        near_dist = self.track_data._center_line_.project(
            Point(car_model_state.pose.position.x, car_model_state.pose.position.y))
        near_pnt_ctr = self.track_data._center_line_.interpolate(near_dist - self.cam_dist_offset)
        target_camera_location_2d = np.array([near_pnt_ctr.x, near_pnt_ctr.y])

        # Calculate camera rotation quaternion based on lookAt yaw with respect to
        # current camera position and car position
        look_at_yaw = utils.get_angle_between_two_points_2d_rad(Point(target_camera_location_2d[0],
                                                                      target_camera_location_2d[1]),
                                                                car_model_state.pose.position)
        cam_quaternion = euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=look_at_yaw)

        camera_model_state.pose.position.x = target_camera_location_2d[0]
        camera_model_state.pose.position.y = target_camera_location_2d[1]
        camera_model_state.pose.position.z = self.cam_fixed_height

        camera_model_state.pose.orientation.x = cam_quaternion[0]
        camera_model_state.pose.orientation.y = cam_quaternion[1]
        camera_model_state.pose.orientation.z = cam_quaternion[2]
        camera_model_state.pose.orientation.w = cam_quaternion[3]
        self.model_state_client(camera_model_state)

        self.last_camera_state = camera_model_state

    def _update(self, car_model_state, delta_time):
        # Calculate target Camera position based on nearest center track from the car.
        # 1. Project the car position to 1-d global distance of track
        # 2. Minus track with offset and get the point of center track
        near_dist = self.track_data._center_line_.project(
            Point(car_model_state.pose.position.x, car_model_state.pose.position.y))
        near_pnt_ctr = self.track_data._center_line_.interpolate(near_dist - self.cam_dist_offset)
        target_camera_location_2d = np.array([near_pnt_ctr.x, near_pnt_ctr.y])

        # Linear interpolate Camera position to target position
        cur_camera_2d_pos = np.array([self.last_camera_state.pose.position.x,
                                      self.last_camera_state.pose.position.y])
        new_cam_pos_2d = utils.lerp(cur_camera_2d_pos, target_camera_location_2d, delta_time * self.damping)

        # Calculate camera rotation quaternion based on lookAt yaw
        look_at_yaw = utils.get_angle_between_two_points_2d_rad(self.last_camera_state.pose.position,
                                                                car_model_state.pose.position)
        cam_quaternion = euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=look_at_yaw)

        # Configure Camera Model State
        camera_model_state = ModelState()
        camera_model_state.model_name = self.topic_name
        camera_model_state.pose.position.x = new_cam_pos_2d[0]
        camera_model_state.pose.position.y = new_cam_pos_2d[1]
        camera_model_state.pose.position.z = self.cam_fixed_height
        camera_model_state.pose.orientation.x = cam_quaternion[0]
        camera_model_state.pose.orientation.y = cam_quaternion[1]
        camera_model_state.pose.orientation.z = cam_quaternion[2]
        camera_model_state.pose.orientation.w = cam_quaternion[3]
        self.model_state_client(camera_model_state)

        self.last_camera_state = camera_model_state
