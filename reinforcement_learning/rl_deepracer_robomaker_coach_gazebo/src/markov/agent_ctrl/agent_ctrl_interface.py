'''This class this defines an interface for how agents need to interact with the
    environment, all concrete classes should abstract away the communication of the
    agent with gazebo
'''
import abc

class AgentCtrlInterface(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def action_space(self):
      '''Returns a read onlu version of the action space so that is can be passed to coach'''
      raise NotImplementedError('Agent control must be able retrieve action space')

    @abc.abstractmethod
    def reset_agent(self):
        '''reset agent to the reset position

        Raises:
            NotImplementedError: agent control must be able to reset agent
        '''
        raise NotImplementedError('Agent control must be able to reset agent')

    @abc.abstractmethod
    def send_action(self, action):
        '''Send the desired action to the agent

        Args:
            action: Interger with the desired action to take

        Raises:
            NotImplementedError: agent control must be able to send action
        '''
        raise NotImplementedError('Agent control must be able to send action')

    @abc.abstractmethod
    def update_agent(self, action):
        '''Update the agent status after action is taken

        Args:
            action: Interger with the desired action to take

        Returns:
            dict: dictionary contains the agent info after desired action is taken
                  with key as agent's name and value as agent's info

        Raises:
            NotImplementedError: agent control must be able to update agent
        '''
        raise NotImplementedError('Agent control must be able to update agent')

    @abc.abstractmethod
    def judge_action(self, agents_info_map):
        '''Returns the reward, done flag, step metrics after action is taken

        Args:
            agents_info_map: dictionary contains all agents info map with key as
                             each agent's name and value as each agent's info

        Returns:
            tuple (int, bool, dict): reward, done flag, and step metrics tuple

        Raises:
            NotImplementedError: Agent control must be able to judge action
        '''
        raise NotImplementedError('Agent control must be able to judge action')

    @abc.abstractmethod
    def finish_episode(self):
        '''Runs all behavior required at the end of the episode, such as uploading
           debug data to S3.
        '''
        raise NotImplementedError('Agent control must be able to properly handle the end of \
                                   an episode')
