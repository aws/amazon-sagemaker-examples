import time
from typing import Union, List

import numpy as np

from rl_coach.base_parameters import Parameters
from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import GoalType, ActionType, EnvResponse, RunPhase
from rl_coach.environments.environment_interface import EnvironmentInterface
from rl_coach.environments.environment import LevelSelection
from rl_coach.spaces import ActionSpace, ObservationSpace, RewardSpace, StateSpace
from rl_coach.utils import force_list


class MultiAgentEnvironmentParameters(Parameters):
    def __init__(self, level=None):
        super().__init__()
        self.level = level
        self.frame_skip = 4
        self.seed = None
        self.custom_reward_threshold = None
        self.default_input_filter = None
        self.default_output_filter = None
        self.experiment_path = None

        # Set target reward and target_success if present
        self.target_success_rate = 1.0

    @property
    def path(self):
        return 'markov.multi_agent_coach.multi_agent_environment:MultiAgentEnvironment'


class MultiAgentEnvironment(EnvironmentInterface):
    def __init__(self, level: LevelSelection, seed: int, frame_skip: int,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 target_success_rate: float=1.0, num_agents: int=1, **kwargs):
        """
        :param level: The environment level. Each environment can have multiple levels
        :param seed: a seed for the random number generator of the environment
        :param frame_skip: number of frames to skip (while repeating the same action) between each two agent directives
        :param visualization_parameters: a blob of parameters used for visualization of the environment
        :param **kwargs: as the class is instantiated by MultiAgentEnvironmentParameters, this is used to support having
                         additional arguments which will be ignored by this class, but might be used by others
        """
        super().__init__()

        # env initialization
        self.num_agents = num_agents
        self.state = [{}] * num_agents
        self.reward = [0.0] * num_agents
        self.done = [False] * num_agents
        self.goal = None
        self.info = {}
        self._last_env_response = [None] * num_agents
        self.last_action = [0] * num_agents
        self.episode_idx = 0
        self.total_steps_counter = 0
        self.current_episode_steps_counter = 0
        self.last_episode_time = time.time()

        # rewards
        self.total_reward_in_current_episode = [0.0] * num_agents
        self.max_reward_achieved = [-np.inf] * num_agents
        self.reward_success_threshold = custom_reward_threshold

        # spaces
        self.state_space = self._state_space = [None] * num_agents
        self.goal_space = self._goal_space = None
        self.action_space = self._action_space = [None] * num_agents
        self.reward_space = RewardSpace(1, reward_success_threshold=self.reward_success_threshold)  # TODO: add a getter and setter

        self.env_id = str(level)
        self.seed = seed
        self.frame_skip = frame_skip

        # visualization
        self.visualization_parameters = visualization_parameters

        # Set target reward and target_success if present
        self.target_success_rate = target_success_rate

    @property
    def action_space(self) -> Union[List[ActionSpace], ActionSpace]:
        """
        Get the action space of the environment

        :return: the action space
        """
        return self._action_space

    @action_space.setter
    def action_space(self, val: Union[List[ActionSpace], ActionSpace]):
        """
        Set the action space of the environment

        :return: None
        """
        self._action_space = val

    @property
    def state_space(self) -> Union[List[StateSpace], StateSpace]:
        """
        Get the state space of the environment

        :return: the observation space
        """
        return self._state_space

    @state_space.setter
    def state_space(self, val: Union[List[StateSpace], StateSpace]):
        """
        Set the state space of the environment

        :return: None
        """
        self._state_space = val

    @property
    def goal_space(self) -> Union[List[ObservationSpace], ObservationSpace]:
        """
        Get the state space of the environment

        :return: the observation space
        """
        return self._goal_space

    @goal_space.setter
    def goal_space(self, val: Union[List[ObservationSpace], ObservationSpace]):
        """
        Set the goal space of the environment

        :return: None
        """
        self._goal_space = val

    @property
    def last_env_response(self) -> Union[List[EnvResponse], EnvResponse]:
        """
        Get the last environment response

        :return: a dictionary that contains the state, reward, etc.
        """
        return self._last_env_response

    @last_env_response.setter
    def last_env_response(self, val: Union[List[EnvResponse], EnvResponse]):
        """
        Set the last environment response

        :param val: the last environment response
        """
        self._last_env_response = force_list(val)

    def step(self, action: Union[List[ActionType], ActionType]) -> List[EnvResponse]:
        """
        Make a single step in the environment using the given action

        :param action: an action to use for stepping the environment. Should follow the definition of the action space.
        :return: the environment response as returned in get_last_env_response
        """
        for agent_action, action_space in zip(force_list(action), force_list(self.action_space)):
            agent_action = action_space.clip_action_to_space(agent_action)
            if action_space and not action_space.contains(agent_action):
                raise ValueError("The given action does not match the action space definition. "
                                 "Action = {}, action space definition = {}".format(agent_action, action_space))

        # store the last agent action done and allow passing None actions to repeat the previously done action
        if action is None:
            action = self.last_action
        self.last_action = action

        self.current_episode_steps_counter += 1
        if self.phase != RunPhase.UNDEFINED:
            self.total_steps_counter += 1

        # act
        self._take_action(action)

        # observe
        self._update_state()

        self.total_reward_in_current_episode = [total_reward_in_current_episode + reward
                                                for total_reward_in_current_episode, reward in
                                                zip(self.total_reward_in_current_episode, self.reward)]

        self.last_env_response = \
            [EnvResponse(
                next_state=state,
                reward=reward,
                game_over=done,
                goal=self.goal,
                info=self.info
            ) for state, reward, done in zip(self.state, self.reward, self.done)]

        return self.last_env_response

    def handle_episode_ended(self) -> None:
        """
        End an episode

        :return: None
        """
        pass

    def reset_internal_state(self, force_environment_reset=False) -> EnvResponse:
        """
        Reset the environment and all the variable of the wrapper

        :param force_environment_reset: forces environment reset even when the game did not end
        :return: A dictionary containing the observation, reward, done flag, action and measurements
        """

        self._restart_environment_episode(force_environment_reset)
        self.last_episode_time = time.time()

        if self.current_episode_steps_counter > 0 and self.phase != RunPhase.UNDEFINED:
            self.episode_idx += 1

        self.done = [False] * self.num_agents
        self.total_reward_in_current_episode = self.reward = [0.0] * self.num_agents
        self.last_action = [0] * self.num_agents
        self.current_episode_steps_counter = 0
        self._update_state()

        self.last_env_response = \
            [EnvResponse(
                next_state=state,
                reward=reward,
                game_over=done,
                goal=self.goal,
                info=self.info
            ) for state, reward, done in zip(self.state, self.reward, self.done)]

        return self.last_env_response

    def get_random_action(self) -> Union[List[ActionType], ActionType]:
        """
        Returns an action picked uniformly from the available actions

        :return: a numpy array with a random action
        """
        if type(self.action_space) == ActionType:
            return self.action_space.sample()
        else:
            return [action_space.sample() for action_space in self.action_space]

    def get_goal(self) -> GoalType:
        """
        Get the current goal that the agents needs to achieve in the environment

        :return: The goal
        """
        return self.goal

    def set_goal(self, goal: GoalType) -> None:
        """
        Set the current goal that the agent needs to achieve in the environment

        :param goal: the goal that needs to be achieved
        :return: None
        """
        self.goal = goal

    # The following functions define the interaction with the environment.
    # Any new environment that inherits the MultiAgentEnvironment class should use these signatures.
    # Some of these functions are optional - please read their description for more details.

    def _take_action(self, action_idx: ActionType) -> None:
        """
        An environment dependent function that sends an action to the simulator.

        :param action_idx: the action to perform on the environment
        :return: None
        """
        raise NotImplementedError("")

    def _update_state(self) -> None:
        """
        Updates the state from the environment.
        Should update self.state, self.reward, self.done, and self.info

        :return: None
        """
        raise NotImplementedError("")

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        """
        Restarts the simulator episode

        :param force_environment_reset: Force the environment to reset even if the episode is not done yet.
        :return: None
        """
        raise NotImplementedError("")

    def get_target_success_rate(self) -> float:
        return self.target_success_rate

    def close(self) -> None:
        """
        Clean up steps.

        :return: None
        """
        pass
