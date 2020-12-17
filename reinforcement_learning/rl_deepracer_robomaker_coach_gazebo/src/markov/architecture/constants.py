'''This module should house all data that is to be constant across the DeepRacer
   simulation application. The module used enums to scope the constants and to
   allow iteration over the constants so that clients can check against the
   available constants.
'''
from enum import Enum

class EmbedderType(Enum):
    '''Enum containing the keys for supported scheme's
    '''
    SCHEME = 'scheme'
    BN_SCHEME = 'bn_scheme'

class SchemeInfo(Enum):
    '''Enum containing the keys required for dictionaries passed into the
       embedder creators for generating schemes. If the signature of the schemes
       changes this enum should be updated.
     '''
    CONV_INFO_LIST = 'conv_info_list'
    DENSE_LAYER_INFO_LIST = 'dense_layer__hidden_unit_list'
    BN_INFO_CONV = 'bn_info_conv'
    BN_INFO_DENSE = 'bn_info_dense'
    IS_FIRST_LAYER_BN = 'is_first_layer_bn'

class Input(Enum):
    '''Enum with available inputs, as we add sensors we should add inputs.
       This is also important for networks with more than one input.
    '''
    OBSERVATION = 'observation'
    LIDAR = 'LIDAR'
    SECTOR_LIDAR = 'SECTOR_LIDAR'
    CAMERA = 'FRONT_FACING_CAMERA'
    LEFT_CAMERA = 'LEFT_CAMERA'
    STEREO = 'STEREO_CAMERAS'
    
    @classmethod
    def validate_inputs(cls, input_list):
        '''Returns True if the all the inputs in input_list is supported, False otherwise'''
        return all([_input in cls._value2member_map_ for _input in input_list])

class ActivationFunctions(Enum):
    '''Enum containing the available activation functions for rl coach.'''
    RELU = 'relu'
    TANH = 'tanh'
    NONE = 'none'

    @classmethod
    def has_activation_function(cls, activation_function):
        '''Returns True if the activation function is supported, False otherwise
            activation_function - String containing activation function to check
        '''
        return activation_function in cls._value2member_map_

class NeuralNetwork(Enum):
    """Enum containing the keys for neural networks"""
    DEEP_CONVOLUTIONAL_NETWORK_SHALLOW = 'DEEP_CONVOLUTIONAL_NETWORK_SHALLOW'
    DEEP_CONVOLUTIONAL_NETWORK = 'DEEP_CONVOLUTIONAL_NETWORK'
    DEEP_CONVOLUTIONAL_NETWORK_DEEP = 'DEEP_CONVOLUTIONAL_NETWORK_DEEP'
