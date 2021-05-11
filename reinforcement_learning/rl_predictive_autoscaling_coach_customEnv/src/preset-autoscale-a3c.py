from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.policy_optimization_agent import PolicyGradientRescaler
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import (
    DistributedCoachSynchronizationType,
    PresetValidationParameters,
    VisualizationParameters,
)
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, RunPhase, TrainingSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.reward.reward_rescale_filter import RewardRescaleFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(int(5e5))
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(50000)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(150000)

#########
# Agent #
#########
agent_params = ActorCriticAgentParameters()

agent_params.algorithm.policy_gradient_rescaler = PolicyGradientRescaler.GAE
agent_params.input_filter = InputFilter()
agent_params.input_filter.add_reward_filter("rescale", RewardRescaleFilter(1 / 10000.0))
agent_params.algorithm.num_steps_between_gradient_updates = 30
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 0.99
agent_params.algorithm.beta_entropy = 0.01

agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.05
agent_params.algorithm.estimate_state_value_using_gae = True
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(2048)

agent_params.network_wrappers["main"].learning_rate = 0.0003
agent_params.network_wrappers["main"].input_embedders_parameters[
    "observation"
].activation_function = "tanh"
agent_params.network_wrappers["main"].input_embedders_parameters["observation"].scheme = [Dense(64)]
agent_params.network_wrappers["main"].middleware_parameters.scheme = [Dense(64)]
agent_params.network_wrappers["main"].middleware_parameters.activation_function = "tanh"
agent_params.network_wrappers["main"].batch_size = 64
agent_params.network_wrappers["main"].optimizer_epsilon = 1e-5
agent_params.network_wrappers["main"].clip_gradients = 40.0

agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)

###############
# Environment #
###############
env_params = GymVectorEnvironment(level="autoscalesim:SimpleScalableWebserviceSim")

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 150
preset_validation_params.max_episodes_to_achieve_reward = 400

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=VisualizationParameters(),
    preset_validation_params=preset_validation_params,
)
