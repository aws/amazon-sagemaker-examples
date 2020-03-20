'''This module contains the concrete implementations of the agent interface'''
from markov.architecture.constants import Input

class Agent(object):
    PAUSE_REWARD = 0.0
    '''Concrete class for agents running in the rollout worker'''
    def __init__(self, network_settings, sensor, ctrl):
        '''network_settings - Dictionary containing the desired
                              network configuration
           sensor - Reference to the composite sensor object
           ctrl - Reference to the car control object
           metrics - Reference to the metrics object
        '''
        self._network_settings_ = network_settings
        self._sensor_ = sensor
        self._ctrl_ = ctrl

    @property
    def network_settings(self):
        return self._network_settings_

    def get_observation_space(self):
        '''Get the sensor obervation space
        '''
        if self._sensor_ is not None:
            return self._sensor_.get_observation_space()

    def get_action_space(self):
        '''Get the control action space
        '''
        return self._ctrl_.action_space

    def reset_agent(self):
        '''Reset agent control and metric instance
        '''
        self._ctrl_.reset_agent()
        if self._sensor_ is not None:
            return self._sensor_.get_state()

    def finish_episode(self):
        ''' Finish episode and update its metrics into s3 bucket
        '''
        self._ctrl_.finish_episode()

    def send_action(self, action):
        '''Publish action index to gazebo

        Args:
            action: Interger with the desired action to take
        '''
        self._ctrl_.send_action(action)

    def update_agent(self, action):
        '''Update agent status based on taken action and env change

        Args:
            action: Interger with the desired action to take

        Returns:
            dict: dictionary contains single agent info map after desired action is taken
                  with key as each agent's name and value as each agent's info
        '''
        return self._ctrl_.update_agent(action)

    def judge_action(self, action, agents_info_map):
        '''Judge the taken action

        Args:
            action: Interger with the desired action to take
            agents_info_map: dictionary contains all agents info map with key as
                             each agent's name and value as each agent's info

        Returns:
            tuple: tuple contains next state sensor observation, reward value, and done flag
        '''
        if self._sensor_ is not None:
            next_state = self._sensor_.get_state()
        else:
            next_state = None
        reward, done, _ = self._ctrl_.judge_action(agents_info_map)
        if hasattr(self._ctrl_, 'reward_data_pub') and self._ctrl_.reward_data_pub is not None:
            raw_state = self._sensor_.get_raw_state()
            # More visualizations topics can be added here
            if Input.CAMERA.value in raw_state and raw_state[Input.CAMERA.value] is not None:
                self._ctrl_.reward_data_pub.publish_frame(raw_state[Input.CAMERA.value], action, reward)
            elif Input.OBSERVATION.value in raw_state and raw_state[Input.OBSERVATION.value] is not None:
                self._ctrl_.reward_data_pub.publish_frame(raw_state[Input.OBSERVATION.value], action, reward)
        return next_state, reward, done
