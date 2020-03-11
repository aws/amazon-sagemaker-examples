import math
import logging
import xml.etree.ElementTree as ET
import rospy

from deepracer_simulation_environment.srv import TopCamDataSrvResponse, TopCamDataSrv
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.track_geom.track_data import TrackData
from markov.track_geom.utils import euler_to_quaternion
from markov.cameras.abs_camera import AbstractCamera
from markov.cameras.constants import CameraSettings
from markov.utils import Logger

# Height value is determined from AWS track and is maintained to prevent z fighting in top down
# view
CAMERA_HEIGHT = 6.0
# Percentage to pad the image so that the frame boundary is not exactly on the track
PADDING_PCT = 0.25
# The default horizontal field of view
DEFAULT_H_FOV = 1.13
# Default resolution
DEFAULT_RESOLUTION = (640, 480)
# Logger object
LOG = Logger(__name__, logging.INFO).get_logger()

class TopCamera(AbstractCamera):
    """this module is for top camera"""
    name = "top_camera"

    def __init__(self, namespace=None, topic_name=None):
        super(TopCamera, self).__init__(TopCamera.name, namespace=namespace,
                                        topic_name=topic_name)
        rospy.wait_for_service('/gazebo/set_model_state')
        self.model_state_client = ServiceProxyWrapper('/gazebo/set_model_state', SetModelState)
        self.track_data = TrackData.get_instance()
        x_min, y_min, x_max, y_max = self.track_data._outer_border_.bounds
        horizontal_width = (x_max - x_min) * (1.0 + PADDING_PCT)
        vertical_width = (y_max - y_min) * (1.0 + PADDING_PCT)
        horizontal_fov = DEFAULT_H_FOV
        try:
            if horizontal_width >= vertical_width:
                horizontal_fov = 2.0 * math.atan(0.5 * horizontal_width / CAMERA_HEIGHT)
            else:
                vertical_fov = math.atan(0.5 * vertical_width / CAMERA_HEIGHT)
                aspect_ratio = float(DEFAULT_RESOLUTION[0]) / float(DEFAULT_RESOLUTION[1])
                horizontal_fov = 2.0 * math.atan(aspect_ratio * math.tan(vertical_fov))
        except Exception as ex:
            LOG.info('Unable to compute top camera fov, using default: %s', ex)

        self.camera_settings_dict = CameraSettings.get_empty_dict()
        self.camera_settings_dict[CameraSettings.HORZ_FOV] = horizontal_fov
        self.camera_settings_dict[CameraSettings.PADDING_PCT] = PADDING_PCT
        self.camera_settings_dict[CameraSettings.IMG_WIDTH] = DEFAULT_RESOLUTION[0]
        self.camera_settings_dict[CameraSettings.IMG_HEIGHT] = DEFAULT_RESOLUTION[1]

        rospy.Service('get_top_cam_data', TopCamDataSrv, self._handle_get_top_cam_data)

    def _handle_get_top_cam_data(self, req):
        '''Response handler for clients requesting the camera settings data
           req - Client request, which should be an empty request
        '''
        return TopCamDataSrvResponse(self.camera_settings_dict[CameraSettings.HORZ_FOV],
                                     self.camera_settings_dict[CameraSettings.PADDING_PCT],
                                     self.camera_settings_dict[CameraSettings.IMG_WIDTH],
                                     self.camera_settings_dict[CameraSettings.IMG_HEIGHT])

    def _get_sdf_string(self, camera_sdf_path):
        tree = ET.parse(camera_sdf_path)
        root = tree.getroot()
        for fov in root.iter('horizontal_fov'):
            fov.text = str(self.camera_settings_dict[CameraSettings.HORZ_FOV])
        return '<?xml version="1.0"?>\n {}'.format(ET.tostring(root))

    def _get_initial_camera_pose(self, model_state):
        # get the bounds
        x_min, y_min, x_max, y_max = self.track_data._outer_border_.bounds
        # update camera position
        model_pose = Pose()
        model_pose.position.x = (x_min+x_max) / 2.0
        model_pose.position.y = (y_min+y_max) / 2.0
        model_pose.position.z = CAMERA_HEIGHT
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
