from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.observation.observation_normalization_filter import ObservationNormalizationFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule
import eplus

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(2000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(1000)
schedule_params.evaluation_steps = EnvironmentEpisodes(2)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
agent_params = ClippedPPOAgentParameters()


agent_params.network_wrappers['main'].learning_rate = 0.0003
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].activation_function = 'tanh'
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(64)]
agent_params.network_wrappers['main'].middleware_parameters.activation_function = 'tanh'
agent_params.network_wrappers['main'].batch_size = 64
agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000)
agent_params.algorithm.beta_entropy = 0
agent_params.algorithm.gae_lambda = 0.95
agent_params.algorithm.discount = 0.99
agent_params.algorithm.optimization_epochs = 10
agent_params.algorithm.estimate_state_value_using_gae = True
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1000)

# Distributed Coach synchronization type.
agent_params.algorithm.distributed_coach_synchronization_type = DistributedCoachSynchronizationType.SYNC

agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 10000)
agent_params.pre_network_filter.add_observation_filter('observation', 'normalize_observation',
    ObservationNormalizationFilter(name='normalize_observation'))

###############
# Environment #
###############
config = {
    'eplus_path': '/usr/local/EnergyPlus-8-8-0/',
    'weather_file': 'weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw'
}
env_params = GymVectorEnvironment(level='eplus.envs.data_center_env:DataCenterEnv')
env_params.additional_simulator_parameters = {'config': config }

#################
# Visualization #
#################

vis_params = VisualizationParameters()
vis_params.dump_gifs = False

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 150
preset_validation_params.max_episodes_to_achieve_reward = 400

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)