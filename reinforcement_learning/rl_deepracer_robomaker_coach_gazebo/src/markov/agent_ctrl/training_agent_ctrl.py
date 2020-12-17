'''This module defines the concrete classes for training'''
from markov.agent_ctrl.utils import load_action_space
from markov.agent_ctrl.agent_ctrl_interface import AgentCtrlInterface

class TrainingCtrl(AgentCtrlInterface):
    '''Concrete class for an agent that drives forward'''
    def __init__(self, agent_name, path_to_json):
        '''agent_name - String containing the name of the agent
           path_to_json - String containing absolute path to model meta data json containing
                          the action space
        '''
        # Store the name of the agent used to set agents position on the track
        self._agent_name_ = agent_name
        #Create default reward parameters
        self._action_space_, _ = load_action_space(path_to_json)

    @property
    def action_space(self):
        return self._action_space_

    def reset_agent(self):
        pass

    def send_action(self, action):
        pass

    def update_agent(self, action):
        return {}

    def judge_action(self, agents_info_map):
        return None, None, None

    def finish_episode(self):
        pass
