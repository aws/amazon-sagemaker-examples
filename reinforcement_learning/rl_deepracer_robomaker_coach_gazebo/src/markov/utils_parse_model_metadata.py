"""utils method to parse sensors and networks from model meta json file"""
import json
import logging
from markov.architecture.constants import Input, NeuralNetwork
from markov.utils import Logger, SIMAPP_VERSION

LOG = Logger(__name__, logging.INFO).get_logger()

def parse_model_metadata(local_model_metadata_path):
    """tuple: return a tuple of (bool, bool, str) for include_second_camera,
       include_lidar_sensor, and network"""
    try:
        with open(local_model_metadata_path, "r") as json_file:
            data = json.load(json_file)
            if 'sensor' in data:
                sensor = data['sensor']
                simapp_version = SIMAPP_VERSION
            else:
                sensor = [Input.OBSERVATION.value]
                simapp_version = "1.0"
            if 'neural_network' in data:
                network = data['neural_network']
            else:
                network = NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value
        LOG.info("Sensor list %s, network %s, simapp_version %s", sensor, network, simapp_version)
        return sensor, network, simapp_version
    except Exception as ex:
        raise Exception('Model metadata does not exist: {}'.format(ex))
