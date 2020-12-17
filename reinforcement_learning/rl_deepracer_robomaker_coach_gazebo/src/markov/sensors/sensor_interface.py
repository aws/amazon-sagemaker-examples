'''This module defines the interface for all sensors'''
import abc

class SensorInterface(object, metaclass=abc.ABCMeta):
    ''' This class defines an interface for sensors, it defines
        the basic functionality required by sensors being used in
        an environment.
    '''
    @abc.abstractmethod
    def get_observation_space(self):
        ''' Return the observation space for this sensor '''
        raise NotImplementedError('Sensor object must configure observation space')

    @abc.abstractmethod
    def get_state(self, block=True):
        ''' Return the sensor state retrieved from the environment
            block - Whether or not to block until sensor data is available
        '''
        raise NotImplementedError('Sensor object must get state')

    @abc.abstractmethod
    def reset(self):
        ''' Reset the sensor data '''
        raise NotImplementedError('Sensor object must reset')

    @abc.abstractmethod
    def get_input_embedders(self, network_type):
        '''Returns a dictionary representing the input embedder for the sensor
           network_type - The type of network for which to return the embedder for
        '''
        raise NotImplementedError('Sensor does not implement input embedder')
