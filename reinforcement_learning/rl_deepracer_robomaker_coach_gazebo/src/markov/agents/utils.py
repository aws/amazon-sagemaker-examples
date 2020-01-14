'''Utility methods for the agent’s module'''
from markov.architecture.constants import EmbedderType, ActivationFunctions, NeuralNetwork, Input
from markov.architecture.custom_architectures import DEFAULT_MIDDLEWARE, VGG_MIDDLEWARE
from markov.sensors.composite_sensor import CompositeSensor

def construct_sensor(observation_list, factory):
    '''Adds the sensors to the composite sensor based on the given observation list
       sensor - Composite sensor
       observation_list - Observation list containg the sensor information base on architecture
       factory - Object containing the sensor factory method to use
    '''
    sensor = CompositeSensor()
    if Input.LEFT_CAMERA.value in observation_list:
        sensor.add_sensor(factory.create_sensor(Input.LEFT_CAMERA.value, {}))
    if Input.STEREO.value in observation_list:
        sensor.add_sensor(factory.create_sensor(Input.STEREO.value, {}))
    if Input.CAMERA.value in observation_list:
        sensor.add_sensor(factory.create_sensor(Input.CAMERA.value, {}))
    if Input.LIDAR.value in observation_list:
        sensor.add_sensor(factory.create_sensor(Input.LIDAR.value, {}))
    if Input.OBSERVATION.value in observation_list:
        sensor.add_sensor(factory.create_sensor(Input.OBSERVATION.value, {}))
    return sensor

def get_network_settings(sensor, network):
    '''Returns a dictionary containing the network information for the agent based on the
       sensor configuration
       netwirk - Sting of desired network topology shallow, deep, deep-deep
    '''
    try:
        is_deep = network == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_DEEP.value
        return {'input_embedders': sensor.get_input_embedders(network),
                'middleware_embedders': VGG_MIDDLEWARE if is_deep else DEFAULT_MIDDLEWARE,
                'embedder_type': EmbedderType.SCHEME.value,
                'activation_function': ActivationFunctions.RELU.value}
    except Exception as ex:
        raise Exception("network: {} failed to load: {}, ".format(network, ex))
