'''This module is used to create agents for the training worker'''
from markov.agent_ctrl.constants import ConfigParams
from markov.agent_ctrl.training_agent_ctrl import TrainingCtrl
from markov.agents.agent import Agent
from markov.agents.utils import construct_sensor, get_network_settings
from markov.sensors.sensors_training import SensorFactory
from markov import utils_parse_model_metadata

def create_training_agent(agent_config):
    '''Returns an training agent object
       agent_config - Dictionary containing the key specified in ConfigParams
    '''
    model_metadata = agent_config['model_metadata']
    observation_list, network, _ = utils_parse_model_metadata.parse_model_metadata(model_metadata)
    sensor = construct_sensor(agent_config[ConfigParams.CAR_CTRL_CONFIG.value][ConfigParams.AGENT_NAME.value], observation_list, SensorFactory)
    network_settings = get_network_settings(sensor, network)

    ctrl_config = agent_config[ConfigParams.CAR_CTRL_CONFIG.value]
    ctrl = TrainingCtrl(ctrl_config[ConfigParams.AGENT_NAME.value],
                        ctrl_config[ConfigParams.ACTION_SPACE_PATH.value])

    return Agent(network_settings, sensor, ctrl)
