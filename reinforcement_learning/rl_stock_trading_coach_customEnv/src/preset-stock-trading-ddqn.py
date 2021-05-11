# Preset file in Amazon SageMaker RL
from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.architectures.head_parameters import DuelingQHeadParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

#################
# Graph Scheduling
#################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(50000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(5000)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(1000)

############
# DQN Agent
############

agent_params = DDQNAgentParameters()

# DQN params
agent_params.algorithm.discount = 0.99
agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)

# NN configuration
agent_params.network_wrappers["main"].batch_size = 32
agent_params.network_wrappers["main"].learning_rate = 0.0001
agent_params.network_wrappers["main"].input_embedders_parameters["observation"].scheme = [
    Dense(512)
]
agent_params.network_wrappers["main"].replace_mse_with_huber_loss = False
agent_params.network_wrappers["main"].heads_parameters = [DuelingQHeadParameters()]
agent_params.network_wrappers["main"].middleware_parameters.scheme = [Dense(512)]

# ER size
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10000)

# E-Greedy schedule
agent_params.exploration.epsilon_schedule = LinearSchedule(1.0, 0.01, 40000)

#############
# Environment
#############

env_params = GymVectorEnvironment(level="trading_env:TradingEnv")

##################
# Manage resources
##################

preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=VisualizationParameters(),
    preset_validation_params=preset_validation_params,
)
