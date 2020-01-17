'''This module defines the interface between rl coach and the agents enviroment'''
from __future__ import print_function
from typing import  List, Union

from rl_coach.base_parameters import AgentParameters, VisualizationParameters
from rl_coach.environments.environment import LevelSelection
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter

from markov.agents.agent import Agent
from markov.multi_agent_coach.multi_agent_environment \
    import MultiAgentEnvironment, MultiAgentEnvironmentParameters
from markov.deepracer_exceptions import GenericRolloutException, GenericTrainerException, \
                                        RewardFunctionError
from markov.utils import log_and_exit, SIMAPP_SIMULATION_WORKER_EXCEPTION, \
                         SIMAPP_EVENT_ERROR_CODE_500

# Max number of steps to allow per episode
MAX_STEPS = 10000

class DeepRacerRacetrackEnvParameters(MultiAgentEnvironmentParameters):
    '''This class defined the environment parameters for DeepRacer, parameters
       added here can be passed to the DeepRacerRacetrackEnv class by adding
       the parameter name to the constructor signature of DeepRacerRacetrackEnv
    '''
    def __init__(self, level=None):
        super().__init__(level=level)
        self.frame_skip = 1
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()
        self.agents_params = None
        self.non_trainable_agents = None

    @property
    def path(self):
        return 'markov.environments.deepracer_racetrack_env:DeepRacerRacetrackEnv'

class DeepRacerRacetrackEnv(MultiAgentEnvironment):
    '''This class defines the mechanics of how a DeepRacer agent interacts
       with the enviroment
    '''
    def __init__(self, level: LevelSelection, seed: int, frame_skip: int,
                 custom_reward_threshold: Union[int, float],
                 visualization_parameters: VisualizationParameters,
                 agents_params: List[AgentParameters],
                 non_trainable_agents: List[Agent],
                 target_success_rate: float = 1.0, **kwargs):
        super().__init__(level, seed, frame_skip,
                         custom_reward_threshold, visualization_parameters,
                         target_success_rate, num_agents=len(agents_params))
        try:
            # Maintain a list of all agents
            self.agent_list = [agent_param.env_agent for agent_param in agents_params]
            self.non_trainable_agents = non_trainable_agents
            # Create the state and action space
            self.state_space = [agent.get_observation_space() for agent in self.agent_list]
            self.action_space = [agent.get_action_space() for agent in self.agent_list]
            # Initialize step count
            self.steps = 0
            # Initialize the state by getting a new state from the environment
            self.reset_internal_state(True)
        except GenericTrainerException as ex:
            ex.log_except_and_exit()
        except GenericRolloutException as ex:
            ex.log_except_and_exit()
        except Exception as ex:
            log_and_exit('Unclassified exception: {}'.format(ex),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def _take_action(self, action_list):
        try:
            # Update state, reward, done flag
            self.state = list()
            self.reward = list()
            self.done = list()
            [agent.step(None) for agent in self.non_trainable_agents]
            for agent, action in zip(self.agent_list, action_list):
                next_state, reward, done = agent.step(action)
                self.state.append(next_state)
                self.reward.append(reward)
                self.done.append(done)
            # Preserve behavior of TimeLimit wrapper
            self.steps += 1
            if MAX_STEPS <= self.steps:
                self.done = [True] * self.num_agents
            # Terminate the episode if any agent is done
            if any(self.done):
                [agent.finish_episode() for agent in self.non_trainable_agents]
                [agent.finish_episode() for agent in self.agent_list]
        except GenericTrainerException as ex:
            ex.log_except_and_exit()
        except GenericRolloutException as ex:
            ex.log_except_and_exit()
        except RewardFunctionError as err:
            err.log_except_and_exit()
        except Exception as ex:
            log_and_exit('Unclassified exception: {}'.format(ex),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)

    def _update_state(self):
        pass

    def _restart_environment_episode(self, force_environment_reset=False):
        try:
            self.steps = 0
            [agent.reset_agent() for agent in self.non_trainable_agents]
            self.state = [agent.reset_agent() for agent in self.agent_list]
            # Reset state, reward, done flag
            self.reward = [0.0] * self.num_agents
            self.done = [False] * self.num_agents
        except GenericTrainerException as ex:
            ex.log_except_and_exit()
        except GenericRolloutException as ex:
            ex.log_except_and_exit()
        except Exception as ex:
            log_and_exit('Unclassified exception: {}'.format(ex),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
