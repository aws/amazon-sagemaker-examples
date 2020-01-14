'''This module contains the available sensors for the sim app'''
import logging
import rospy
import numpy as np
from sensor_msgs.msg import Image as sensor_image
from sensor_msgs.msg import LaserScan
from PIL import Image
from markov.sensors.utils import get_observation_space, get_front_camera_embedders, \
                                 get_left_camera_embedders, get_stereo_camera_embedders, \
                                 get_lidar_embedders, get_observation_embedder
from markov.sensors.sensor_interface import SensorInterface
from markov.environments.constants import TRAINING_IMAGE_SIZE
from markov.architecture.constants import Input
from markov.deepracer_exceptions import GenericRolloutException
from markov import utils

LOGGER = utils.Logger(__name__, logging.INFO).get_logger()

class SensorFactory(object):
    '''This class implements a sensot factory and is used to create sensors per
       agent.
    '''
    @staticmethod
    def create_sensor(sensor_type, config_dict):
        '''Factory method for creating sensors
            type - String containing the desired sensor type
            kwargs - Meta data, usually containing the topics to subscribe to, the
                     concrete sensor classes are responsible for checking the topics.
        '''
        if sensor_type == Input.CAMERA.value:
            return Camera()
        elif sensor_type == Input.LEFT_CAMERA.value:
            return LeftCamera()
        elif sensor_type == Input.STEREO.value:
            return DualCamera()
        elif sensor_type == Input.LIDAR.value:
            return Lidar()
        elif sensor_type == Input.OBSERVATION.value:
            return Observation()
        else:
            raise GenericRolloutException("Unknown sensor")

class Camera(SensorInterface):
    '''Single camera sensor'''
    def __init__(self):
        self.image_buffer = utils.DoubleBuffer()
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self._camera_cb_)
        self.raw_data = None
        self.sensor_type = Input.CAMERA.value

    def get_observation_space(self):
        try:
            return get_observation_space(Input.CAMERA.value)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def get_state(self, block=True):
        try:
            # Make sure the first image is the starting image
            image_data = self.image_buffer.get(block=block)
            # Read the image and resize to get the state
            image = Image.frombytes('RGB', (image_data.width, image_data.height),
                                    image_data.data, 'raw', 'RGB', 0, 1)
            image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
            self.raw_data = image_data
            return {Input.CAMERA.value: np.array(image)}
        except utils.DoubleBuffer.Empty:
            return {}
        except Exception as ex:
            raise GenericRolloutException("Unable to set state: {}".format(ex))

    def reset(self):
        self.image_buffer.clear()

    def get_input_embedders(self, network_type):
        try:
            return get_front_camera_embedders(network_type)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def _camera_cb_(self, data):
        ''' Callback for the single camera, this is triggered by ROS
            data - Image data from the gazebo plugin, it is a sensor message
        '''
        try:
            self.image_buffer.put(data)
        except Exception as ex:
            LOGGER.info("Unable to retrieve frame: %s", ex)

class Observation(SensorInterface):
    '''Single camera sensor that is compatible with simapp v1'''
    def __init__(self):
        self.image_buffer = utils.DoubleBuffer()
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self._camera_cb_)
        self.sensor_type = Input.OBSERVATION.value
        self.raw_data = None
    
    def get_observation_space(self):
        try:
            return get_observation_space(Input.OBSERVATION.value)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def get_state(self, block=True):
        try:
            # Make sure the first image is the starting image
            image_data = self.image_buffer.get(block=block)
            # Read the image and resize to get the state
            image = Image.frombytes('RGB', (image_data.width, image_data.height),
                                    image_data.data, 'raw', 'RGB', 0, 1)
            image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
            self.raw_data = image_data
            return {Input.OBSERVATION.value: np.array(image)}
        except utils.DoubleBuffer.Empty:
            return {}
        except Exception as ex:
            raise GenericRolloutException("Unable to set state: {}".format(ex))

    def reset(self):
        self.image_buffer.clear()

    def get_input_embedders(self, network_type):
        try:
            return get_observation_embedder()
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def _camera_cb_(self, data):
        ''' Callback for the single camera, this is triggered by ROS
            data - Image data from the gazebo plugin, it is a sensor message
        '''
        try:
            self.image_buffer.put(data)
        except Exception as ex:
            LOGGER.info("Unable to retrieve frame: %s", ex)

class LeftCamera(SensorInterface):
    '''This class is specific to left camera's only, it used the same topic as
       the camera class but has a different observation space. If this changes in
       the future this class should be updated.
    '''
    def __init__(self):
        self.image_buffer = utils.DoubleBuffer()
        rospy.Subscriber('/camera/zed/rgb/image_rect_color', sensor_image, self._camera_cb_)
        self.sensor_type = Input.LEFT_CAMERA.value
        self.raw_data = None

    def get_observation_space(self):
        try:
            return get_observation_space(Input.LEFT_CAMERA.value)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def get_state(self, block=True):
        try:
            # Make sure the first image is the starting image
            image_data = self.image_buffer.get(block=block)
            # Read the image and resize to get the state
            image = Image.frombytes('RGB', (image_data.width, image_data.height),
                                    image_data.data, 'raw', 'RGB', 0, 1)
            image = image.resize(TRAINING_IMAGE_SIZE, resample=2)
            self.raw_data = image_data
            return {Input.LEFT_CAMERA.value: np.array(image)}
        except utils.DoubleBuffer.Empty:
            return {}
        except Exception as ex:
            raise GenericRolloutException("Unable to set state: {}".format(ex))

    def reset(self):
        self.image_buffer.clear()

    def get_input_embedders(self, network_type):
        try:
            return get_left_camera_embedders(network_type)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def _camera_cb_(self, data):
        ''' Callback for the single camera, this is triggered by ROS
            data - Image data from the gazebo plugin, it is a sensor message
        '''
        try:
            self.image_buffer.put(data)
        except Exception as ex:
            LOGGER.info("Unable to retrieve frame: %s", ex)

class DualCamera(SensorInterface):
    '''This class handles the data for dual cameras'''
    def __init__(self):
        # Queue used to maintain image consumption synchronicity
        self.image_buffer_left = utils.DoubleBuffer()
        self.image_buffer_right = utils.DoubleBuffer()
        # Set up the subscribers
        rospy.Subscriber('/camera/zed/rgb/image_rect_color',
                         sensor_image,
                         self._left_camera_cb_)
        rospy.Subscriber('/camera/zed_right/rgb/image_rect_color_right',
                         sensor_image,
                         self._right_camera_cb_)
        self.sensor_type = Input.STEREO.value

    def get_observation_space(self):
        try:
            return get_observation_space(Input.STEREO.value)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def get_state(self, block=True):
        try:
            image_data = self.image_buffer_left.get(block=block)
            left_img = Image.frombytes('RGB', (image_data.width, image_data.height),
                                       image_data.data, 'raw', 'RGB', 0, 1)
            left_img = left_img.resize(TRAINING_IMAGE_SIZE, resample=2).convert('L')

            image_data = self.image_buffer_right.get(block=block)
            right_img = Image.frombytes('RGB', (image_data.width, image_data.height),
                                        image_data.data, 'raw', 'RGB', 0, 1)
            right_img = right_img.resize(TRAINING_IMAGE_SIZE, resample=2).convert('L')

            return {Input.STEREO.value: np.array(np.stack((left_img, right_img), axis=2))}
        except utils.DoubleBuffer.Empty:
            return {}
        except Exception as ex:
            raise GenericRolloutException("Unable to set state: {}".format(ex))

    def reset(self):
        self.image_buffer_left.clear()
        self.image_buffer_right.clear()

    def get_input_embedders(self, network_type):
        try:
            return get_stereo_camera_embedders(network_type)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def _left_camera_cb_(self, data):
        ''' Callback for the left camera, this is triggered by ROS
            data - Image data from the gazebo plugin, it is a sensor message
        '''
        try:
            self.image_buffer_left.put(data)
        except Exception as ex:
            LOGGER.info("Unable to retrieve frame: %s", ex)

    def _right_camera_cb_(self, data):
        ''' Callback for the right camera, this is triggered by ROS
            data - Image data from the gazebo plugin, it is a sensor message
        '''
        try:
            self.image_buffer_right.put(data)
        except Exception as ex:
            LOGGER.info("Unable to retrieve frame: %s", ex)

class Lidar(SensorInterface):
    '''This class handles the data collection for lidar'''
    def __init__(self):
        self.data_buffer = utils.DoubleBuffer()
        rospy.Subscriber('/scan', LaserScan, self._scan_cb)
        self.sensor_type = Input.LIDAR.value

    def get_observation_space(self):
        try:
            return get_observation_space(Input.LIDAR.value)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def get_state(self, block=True):
        try:
            return {Input.LIDAR.value: self.data_buffer.get(block=block)}
        except utils.DoubleBuffer.Empty:
            return {}
        except Exception as ex:
            raise GenericRolloutException("Unable to set state: {}".format(ex))

    def reset(self):
        self.data_buffer.clear()

    def get_input_embedders(self, network_type):
        try:
            return get_lidar_embedders(network_type)
        except Exception as ex:
            raise GenericRolloutException('{}'.format(ex))

    def _scan_cb(self, data):
        try:
            self.data_buffer.put(np.array(data.ranges))
        except Exception as ex:
            LOGGER.info("Unable to retrieve state: %s", ex)
