from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import (
    EmbedderScheme,
    MiddlewareScheme,
    PresetValidationParameters,
    VisualizationParameters,
)
from rl_coach.core_types import (
    EnvironmentEpisodes,
    EnvironmentSteps,
    MaxDumpFilter,
    RunPhase,
    SelectedPhaseOnlyDumpFilter,
    TrainingSteps,
)
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(100000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(50000)

#########
# Agent #
#########
agent_params = DQNAgentParameters()

# DQN params
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(500)
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

# NN configuration
agent_params.network_wrappers["main"].learning_rate = 0.00025
agent_params.network_wrappers["main"].replace_mse_with_huber_loss = False
agent_params.network_wrappers["main"].input_embedders_parameters[
    "observation"
].scheme = EmbedderScheme.Shallow
agent_params.network_wrappers["main"].middleware_parameters.scheme = MiddlewareScheme.Shallow

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 50000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.05, 100000)

################
#  Environment #
################
env_params = GymVectorEnvironment(level="patient_envs:PatientMountainCar")

#################
# Visualization #
#################
vis_params = VisualizationParameters()
vis_params.dump_gifs = True
vis_params.video_dump_filters = [SelectedPhaseOnlyDumpFilter(RunPhase.TEST), MaxDumpFilter()]

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = -200
preset_validation_params.max_episodes_to_achieve_reward = 125

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=vis_params,
    preset_validation_params=preset_validation_params,
)
