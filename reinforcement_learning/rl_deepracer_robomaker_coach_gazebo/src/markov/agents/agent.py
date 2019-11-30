from markov.architecture.constants import Input
'''This module contains the concrete implementations of the agent interface'''
class Agent(object):
    '''Concrete class for agents running in the rollout worker'''
    def __init__(self, network_settings, sensor, ctrl, metrics):
        '''network_settings - Dictionary containing the desired
                              network configuration
           sensor - Reference to the composite sensor object
           ctrl - Reference to the car control object
           metrics - Reference to the metrics object
        '''
        self._network_settings_ = network_settings
        self._sensor_ = sensor
        self._ctrl_ = ctrl
        self._metrics_ = metrics

    @property
    def network_settings(self):
        return self._network_settings_

    def get_observation_space(self):
        if self._sensor_ is not None:
            return self._sensor_.get_observation_space()
        else:
            return None

    def get_action_space(self):
        return self._ctrl_.action_space

    def reset_agent(self):
        self._ctrl_.clear_data()
        self._ctrl_.reset_agent()
        if self._metrics_ is not None:
            self._metrics_.reset()
        if self._sensor_ is not None:
            return self._sensor_.get_state()
        else:
            return None

    def finish_episode(self):
        self._ctrl_.finish_episode()
        if self._metrics_:
            self._metrics_.upload_episode_metrics()

    def step(self, action):
        self._ctrl_.send_action(action)
        if self._sensor_ is not None:
            next_state = self._sensor_.get_state()
        else:
            next_state = None
        reward, done, step_metrics = self._ctrl_.judge_action(action)
        if self._metrics_ is not None:
            self._metrics_.upload_step_metrics(step_metrics)
        if hasattr(self._ctrl_, 'reward_data_pub') and self._ctrl_.reward_data_pub is not None:
            raw_state = self._sensor_.get_raw_state()
            # More visualizations topics can be added here
            if Input.CAMERA.value in raw_state and raw_state[Input.CAMERA.value] is not None:
                self._ctrl_.reward_data_pub.publish_frame(raw_state[Input.CAMERA.value], action, reward)
            elif Input.OBSERVATION.value in raw_state and raw_state[Input.OBSERVATION.value] is not None:
                self._ctrl_.reward_data_pub.publish_frame(raw_state[Input.OBSERVATION.value], action, reward)
        return next_state, reward, done
