'''This module is used to create agents for the rollout worker'''
from markov.agent_ctrl.bot_cars_agent_ctrl import  BotCarsCtrl
from markov.agent_ctrl.constants import ConfigParams
from markov.agent_ctrl.rollout_agent_ctrl import RolloutCtrl
from markov.agent_ctrl.obstacles_agent_ctrl import ObstaclesCtrl
from markov.agents.agent import Agent
from markov.agents.utils import construct_sensor, get_network_settings
from markov.sensors.sensors_rollout import SensorFactory
from markov.cameras.frustum_manager import FrustumManager
from markov import utils_parse_model_metadata

def create_rollout_agent(agent_config, metrics, run_phase_subject):
    '''Returns an rollout agent object
       agent_config - Dictionary containing the key specified in ConfigParams
       metrics - Metrics object for the agent
       run_phase_subject - Subject that notifies observers when the run phase changes
    '''
    model_metadata = agent_config['model_metadata']
    observation_list, network, _ = utils_parse_model_metadata.parse_model_metadata(model_metadata)
    agent_name = agent_config[ConfigParams.CAR_CTRL_CONFIG.value][ConfigParams.AGENT_NAME.value]
    sensor = construct_sensor(agent_name, observation_list, SensorFactory)
    network_settings = get_network_settings(sensor, network)
    FrustumManager.get_instance().add(agent_name=agent_name,
                                      observation_list=observation_list)

    ctrl_config = agent_config[ConfigParams.CAR_CTRL_CONFIG.value]
    ctrl = RolloutCtrl(ctrl_config, run_phase_subject, metrics)

    return Agent(network_settings, sensor, ctrl)

def create_obstacles_agent():
    '''Returns an obstacle agent, such as a box. Will not be used for training'''
    ctrl = ObstaclesCtrl()
    return Agent(None, None, ctrl)

def create_bot_cars_agent():
    '''Returns a bot car agent. Will not be used for training'''
    ctrl = BotCarsCtrl()
    return Agent(None, None, ctrl)
