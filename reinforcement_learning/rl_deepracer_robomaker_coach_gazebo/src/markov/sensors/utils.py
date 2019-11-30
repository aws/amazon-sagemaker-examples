'''This module houses all utility methods for the sensor module'''
import numpy as np

from markov.environments.constants import TRAINING_IMAGE_SIZE
from markov.architecture.constants import SchemeInfo, Input, ActivationFunctions, NeuralNetwork
from rl_coach.spaces import StateSpace, ImageObservationSpace, \
                            VectorObservationSpace, PlanarMapsObservationSpace

def get_observation_space(sensor):
    '''Creates the observation space for the given sensor
       sensor - String with the desired sensor to add to the
                observation space
    '''
    obs = StateSpace({})
    if sensor == Input.CAMERA.value or sensor == Input.OBSERVATION.value or \
    sensor == Input.LEFT_CAMERA.value:
        obs[sensor] = ImageObservationSpace(shape=np.array((TRAINING_IMAGE_SIZE[1],
                                                            TRAINING_IMAGE_SIZE[0],
                                                            3)),
                                            high=255,
                                            channels_axis=-1)
    elif sensor == Input.STEREO.value:
        obs[sensor] = PlanarMapsObservationSpace(shape=np.array((TRAINING_IMAGE_SIZE[1],
                                                                 TRAINING_IMAGE_SIZE[0],
                                                                 2)),
                                                 low=0,
                                                 high=255,
                                                 channels_axis=-1)
    elif sensor == Input.LIDAR.value:
        obs[sensor] = VectorObservationSpace(shape=64, low=0.15, high=1.0)
    else:
        raise Exception("Unable to set observation space for sensor {}".format(sensor))
    return obs

#! TODO currently left and front camera use the same embedders, this is how it is wired up
# in custom architectures, decide if this is the best way forward based on current experiments
def get_front_camera_embedders(network_type):
    '''Utility method for retrieving the input embedder for the front camera sensor, this
       needs to be in the util module due to the sagemaker/robomaker incompatibility
       network_type - The type of network for which to return the embedder for
    '''
    input_embedder = dict()
    if network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value:
        input_embedder = {Input.CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    elif network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK.value:
        input_embedder = {Input.CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 5, 2], [32, 3, 1],
                                                             [64, 3, 2], [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [64],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.TANH.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.TANH.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    elif network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_DEEP.value:
        input_embedder = {Input.CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [32, 4, 2],
                                                             [64, 4, 2], [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                           SchemeInfo.BN_INFO_CONV.value: [True, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.5],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    else:
        raise Exception("Camera sensor has no embedder for topology {}".format(network_type))
    return input_embedder

def get_observation_embedder():
    '''Input embedders for the v1.0 simapp'''
    return {Input.OBSERVATION.value:
            {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
             SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
             SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
             SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value, 0.0],
             SchemeInfo.IS_FIRST_LAYER_BN.value: False}}

def get_left_camera_embedders(network_type):
    '''Utility method for retrieving the input embedder for the left camera sensor, this
       needs to be in the util module due to the sagemaker/robomaker incompatibility
       network_type - The type of network for which to return the embedder for
    '''
    input_embedder = dict()
    if network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value:
        input_embedder = {Input.LEFT_CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    elif network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK.value:
        input_embedder = {Input.LEFT_CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 5, 2], [32, 3, 1], [64, 3, 2],
                                                             [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [64],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.TANH.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.TANH.value,
                                                            0.3],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    elif network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_DEEP.value:
        input_embedder = {Input.LEFT_CAMERA.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [32, 4, 2], [64, 4, 2],
                                                             [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [512, 512],
                           SchemeInfo.BN_INFO_CONV.value: [True, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    else:
        raise Exception("Left camera sensor has no embedder for topology {}".format(network_type))
    return input_embedder

def get_stereo_camera_embedders(network_type):
    '''Utility method for retrieving the input embedder for the stereo camera sensor, this
       needs to be in the util module due to the sagemaker/robomaker incompatibility
       network_type - The type of network for which to return the embedder for
    '''
    input_embedder = dict()
    if network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value:
        input_embedder = {Input.STEREO.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 8, 4], [64, 4, 2], [64, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    #! TODO decide if we want to have a deep-deep topology that differes from deep
    elif network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK.value \
        or network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_DEEP.value:
        input_embedder = {Input.STEREO.value:
                          {SchemeInfo.CONV_INFO_LIST.value: [[32, 3, 1], [64, 3, 2], [64, 3, 1],
                                                             [128, 3, 2], [128, 3, 1]],
                           SchemeInfo.DENSE_LAYER_INFO_LIST.value: [],
                           SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value,
                                                           0.0],
                           SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                            0.0],
                           SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    else:
        raise Exception("Stereo camera sensor has no embedder for topology {}".format(network_type))
    return input_embedder

def get_lidar_embedders(network_type):
    '''Utility method for retrieving the input embedder for the lidar camera sensor, this
       needs to be in the util module due to the sagemaker/robomaker incompatibility
       network_type - The type of network for which to return the embedder for
    '''
    #! TODO decide whether we need lidar layers for different network types
    input_embedder = {Input.LIDAR.value:
                      {SchemeInfo.CONV_INFO_LIST.value: [],
                       SchemeInfo.DENSE_LAYER_INFO_LIST.value: [256, 256],
                       SchemeInfo.BN_INFO_CONV.value: [False, ActivationFunctions.RELU.value, 0.0],
                       SchemeInfo.BN_INFO_DENSE.value: [False, ActivationFunctions.RELU.value,
                                                        0.0],
                       SchemeInfo.IS_FIRST_LAYER_BN.value: False}}
    return input_embedder
