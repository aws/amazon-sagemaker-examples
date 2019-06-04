from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.architectures.layers import Dense, Conv2d
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.base_parameters import MiddlewareScheme, DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach.environments.gym_environment import GymVectorEnvironment, ObservationSpaceType
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(250000)
schedule_params.steps_between_evaluation_periods = EnvironmentSteps(2048)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########

agent_params = ActorCriticAgentParameters()

#################
# Visualization #
#################

vis_params = VisualizationParameters()
vis_params.dump_gifs = True

###############
# Environment #
###############

env_params = GymVectorEnvironment(level='tic_tac_toe:TicTacToeEnv')

########
# Test #
########

preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
