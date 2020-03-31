'''This module contains the available sensors for the sim app'''
from markov.architecture.constants import Input
from markov.sensors.sensor_interface import SensorInterface
from markov.sensors.utils import get_observation_space, get_front_camera_embedders, \
                                 get_left_camera_embedders, get_stereo_camera_embedders, \
                                 get_lidar_embedders, get_observation_embedder
from markov.deepracer_exceptions import GenericTrainerException, GenericError
from markov.utils import SIMAPP_TRAINING_WORKER_EXCEPTION

class SensorFactory(object):
    '''This class implements a sensot factory and is used to create sensors per
       agent.
    '''
    @staticmethod
    def create_sensor(racecar_name, sensor_type, config_dict):
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
        elif sensor_type == Input.SECTOR_LIDAR.value:
            return SectorLidar()
        elif sensor_type == Input.OBSERVATION.value:
            return Observation()
        else:
            raise GenericTrainerException("Unknown sensor")

class Camera(SensorInterface):
    '''Single camera sensor'''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.CAMERA.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_front_camera_embedders(network_type)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

class Observation(SensorInterface):
    '''Single camera sensor that is compatible with simapp v1'''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.OBSERVATION.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_observation_embedder()
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

class LeftCamera(SensorInterface):
    '''This class is specific to left camera's only, it used the same topic as
       the camera class but has a different observation space. If this changes in
       the future this class should be updated.
    '''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.LEFT_CAMERA.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_left_camera_embedders(network_type)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

class DualCamera(SensorInterface):
    '''This class handles the data for dual cameras'''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.STEREO.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_stereo_camera_embedders(network_type)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

class Lidar(SensorInterface):
    '''This class handles the data collection for lidar'''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.LIDAR.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_lidar_embedders(network_type, Input.LIDAR.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

class SectorLidar(SensorInterface):
    '''This class handles the data collection for lidar'''
    def get_observation_space(self):
        try:
            return get_observation_space(Input.SECTOR_LIDAR.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))

    def get_state(self, block=True):
        return dict()

    def reset(self):
        pass

    def get_input_embedders(self, network_type):
        try:
            return get_lidar_embedders(network_type, Input.SECTOR_LIDAR.value)
        except GenericError as ex:
            ex.log_except_and_exit(SIMAPP_TRAINING_WORKER_EXCEPTION)
        except Exception as ex:
            raise GenericTrainerException('{}'.format(ex))
