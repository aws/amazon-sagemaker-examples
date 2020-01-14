import numpy as np
import rospy
import math

from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState, ModelStates
from geometry_msgs.msg import Pose
from markov.deepracer_exceptions import GenericRolloutException
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.track_data import TrackData
from markov.cameras import utils
from markov.track_geom.utils import euler_to_quaternion, quaternion_to_euler, apply_orientation
from markov.cameras.abs_camera import BaseCamera


class FollowCarCamera(BaseCamera):
    """this module is for follow-car camera singelton"""
    _instance_ = None
    name = "follow_car_camera"

    @staticmethod
    def get_instance():
        """Method for geting a reference to the camera object"""
        if FollowCarCamera._instance_ is None:
            FollowCarCamera()
        return FollowCarCamera._instance_

    def __init__(self):
        if FollowCarCamera._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple follow car camera")
        super(FollowCarCamera, self).__init__(FollowCarCamera.name)
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
        FollowCarCamera._instance_ = self

    def _get_initial_camera_pose(self, car_model_state):
        _, _, car_yaw = quaternion_to_euler(x=car_model_state.pose.orientation.x,
                                            y=car_model_state.pose.orientation.y,
                                            z=car_model_state.pose.orientation.z,
                                            w=car_model_state.pose.orientation.w)
        target_cam_quaternion = np.array(euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=car_yaw))
        target_camera_location = np.array([car_model_state.pose.position.x,
                                           car_model_state.pose.position.y,
                                           0.0]) + \
                                 apply_orientation(target_cam_quaternion, np.array([-self.cam_dist_offset, 0, 0]))
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

    def _reset(self, car_model_state):
        camera_model_pose = self._get_initial_camera_pose(car_model_state)
        camera_model_state = ModelState()
        camera_model_state.model_name = self.topic_name
        camera_model_state.pose = camera_model_pose
        self.last_camera_state = camera_model_state
        self.model_state_client(camera_model_state)

    def _update(self, car_model_state, delta_time):
        # Calculate target Camera position based on car position
        _, _, car_yaw = quaternion_to_euler(x=car_model_state.pose.orientation.x,
                                            y=car_model_state.pose.orientation.y,
                                            z=car_model_state.pose.orientation.z,
                                            w=car_model_state.pose.orientation.w)
        # Linear Interpolate Yaw angle
        car_yaw = utils.lerp_angle_rad(self.last_yaw, car_yaw, delta_time * self.damping)
        target_cam_quaternion = np.array(euler_to_quaternion(pitch=self.look_down_angle_rad, yaw=car_yaw))
        target_camera_location = np.array([car_model_state.pose.position.x,
                                           car_model_state.pose.position.y,
                                           0.0]) + \
                                 apply_orientation(target_cam_quaternion, np.array([-self.cam_dist_offset, 0, 0]))
        target_camera_location_2d = target_camera_location[0:2]

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
        self.last_yaw = car_yaw
