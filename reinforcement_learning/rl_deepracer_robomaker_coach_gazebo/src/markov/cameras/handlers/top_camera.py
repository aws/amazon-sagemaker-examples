import os
import rospkg
import rospy

from markov.deepracer_exceptions import GenericRolloutException
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.track_data import TrackData
from markov.track_geom.utils import euler_to_quaternion
from markov.cameras.abs_camera import BaseCamera


class TopCamera(BaseCamera):
    """this module is for top camera singelton"""
    _instance_ = None
    name = "top_camera"

    @staticmethod
    def get_instance():
        """Method for geting a reference to the camera object"""
        if TopCamera._instance_ is None:
            TopCamera()
        return TopCamera._instance_

    def __init__(self):
        if TopCamera._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple top video camera")
        super(TopCamera, self).__init__(TopCamera.name)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.model_state_client = ServiceProxyWrapper('/gazebo/set_model_state', SetModelState)
        self.track_data = TrackData.get_instance()
        # there should be only one top video camera
        TopCamera._instance_ = self

    def _get_initial_camera_pose(self, model_state):
        # get the bounds
        x_min, y_min, x_max, y_max = self.track_data._outer_border_.bounds
        # update camera position
        model_pose = Pose()
        model_pose.position.x = (x_min+x_max) / 2.0
        model_pose.position.y = (y_min+y_max) / 2.0
        model_pose.position.z = (x_max-x_min) * 0.9
        x, y, z, w = euler_to_quaternion(roll=1.57079, pitch=1.57079, yaw=3.14159)
        model_pose.orientation.x = x
        model_pose.orientation.y = y
        model_pose.orientation.z = z
        model_pose.orientation.w = w
        return model_pose

    def _reset(self, model_state):
        """Reset camera position based on the track size"""
        if self.is_reset_called:
            return
        # update camera position
        model_pose = self._get_initial_camera_pose(model_state)
        camera_model_state = ModelState()
        camera_model_state.model_name = self.topic_name
        camera_model_state.pose = model_pose
        self.model_state_client(camera_model_state)

    def _update(self, model_state, delta_time):
        pass
