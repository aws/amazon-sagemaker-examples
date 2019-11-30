'''This module is used to create agents for the rollout worker'''
from markov.agent_ctrl.bot_cars_agent_ctrl import  BotCarsCtrl
from markov.agent_ctrl.rollout_agent_ctrl import RolloutCtrl
from markov.agent_ctrl.obstacles_agent_ctrl import ObstaclesCtrl
from markov.agents.agent import Agent
from markov.agents.utils import construct_sensor, get_network_settings
from markov.sensors.sensors_rollout import SensorFactory
from markov.cameras.frustum import Frustum
from markov import utils_parse_model_metadata


def create_rollout_agent(agent_config, metrics):
    '''Returns an rollout agent object
       agent_config - Dictionary containing the key specified in ConfigParams
       metrics - Metrics object for the agent
    '''
    model_metadata = agent_config['model_metadata']
    observation_list, network, _ = utils_parse_model_metadata.parse_model_metadata(model_metadata)
    sensor = construct_sensor(observation_list, SensorFactory)
    network_settings = get_network_settings(sensor, network)
    frustum = Frustum.get_instance()
    frustum.add_cameras(observation_list)

    ctrl_config = agent_config['car_ctrl_cnfig']
    ctrl = RolloutCtrl(ctrl_config)

    return Agent(network_settings, sensor, ctrl, metrics)

def create_obstacles_agent():
    '''Returns an obstacle agent, such as a box. Will not be used for training'''
    ctrl = ObstaclesCtrl()
    return Agent(None, None, ctrl, None)

def create_bot_cars_agent():
    '''Returns a bot car agent. Will not be used for training'''
    ctrl = BotCarsCtrl()
    return Agent(None, None, ctrl, None)
