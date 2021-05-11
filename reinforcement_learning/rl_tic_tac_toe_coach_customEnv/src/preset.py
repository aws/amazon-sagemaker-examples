from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(50000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(2000)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########

agent_params = ClippedPPOAgentParameters()
agent_params.network_wrappers["main"].middleware_parameters.activation_function = "relu"
agent_params.network_wrappers["main"].input_embedders_parameters[
    "observation"
].activation_function = "relu"
agent_params.network_wrappers["main"].learning_rate = 0.001

###############
# Environment #
###############

env_params = GymVectorEnvironment(level="tic_tac_toe:TicTacToeEnv")

########
# Test #
########

preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True

graph_manager = BasicRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=VisualizationParameters(),
    preset_validation_params=preset_validation_params,
)
