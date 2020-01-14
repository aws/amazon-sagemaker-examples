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
        '''Reset the agent to a desired start postion
        '''
        raise NotImplementedError('Agent control must be able to reset agent')

    @abc.abstractmethod
    def send_action(self, action):
        '''Send the desired action to the agent
           action - Integer with the desired action to take
        '''
        raise NotImplementedError('Agent control must be able to send action')

    @abc.abstractmethod
    def judge_action(self, action):
        '''Returns the reward and done flag after and agent takes the action prescribed by a given
           policy
           action - Integer with the desired action to take
        '''
        raise NotImplementedError('Agent control must be able to judge action')

    @abc.abstractmethod
    def finish_episode(self):
        '''Runs all behavior required at the end of the episode, such as uploading
           debug data to S3.
        '''
        raise NotImplementedError('Agent control must be able to properly handle the end of \
                                   an episode')

    @abc.abstractmethod
    def clear_data(self):
      '''Clears the agent data'''
      raise NotImplementedError('Agent control must be able to clear data')
