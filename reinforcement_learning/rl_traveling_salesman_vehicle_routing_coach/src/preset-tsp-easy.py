from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.core_types import *
#from rl_coach.environments.environment import SelectedPhaseOnlyDumpMethod, MaxDumpMethod
from rl_coach import logger
from rl_coach.base_parameters import TaskParameters
from rl_coach.base_parameters import VisualizationParameters

################
#  Environment #
################

env_params = GymVectorEnvironment(level='TSP_env:TSPEasyEnv')

#########
# Agent #
#########

agent_params = ClippedPPOAgentParameters()

#################
# Visualization #
#################

env_params.frame_skip = 5 #to make sure the gifs work without skipping steps

vis_params = VisualizationParameters()
vis_params.dump_gifs=True
#vis_params.video_dump_methods = [SelectedPhaseOnlyDumpMethod(RunPhase.TEST), MaxDumpMethod()]

#experiment_name = "TSPEasy"
#experiment_name = logger.get_experiment_name(experiment_name)
#experiment_path = logger.get_experiment_path(experiment_name)

#task_params = TaskParameters(experiment_path=experiment_path)

####################
# Graph Scheduling #
####################

schedule_params=SimpleSchedule()
schedule_params.improve_steps = TrainingSteps(100000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(20)
schedule_params.evaluation_steps = EnvironmentEpisodes(5)
schedule_params.heatup_steps = EnvironmentSteps(1000)

graph_manager = BasicRLGraphManager(
    agent_params= agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=vis_params
    )

#graph_manager = graph_manager.create_graph(task_parameters=task_params)

#graph_manager.improve()
