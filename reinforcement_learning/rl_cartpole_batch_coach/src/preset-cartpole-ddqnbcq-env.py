from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.agents.ddqn_bcq_agent import DDQNBCQAgentParameters, KNNParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import (
    DistributedCoachSynchronizationType,
    PresetValidationParameters,
    VisualizationParameters,
)
from rl_coach.core_types import (
    CsvDataset,
    EnvironmentEpisodes,
    EnvironmentSteps,
    RunPhase,
    TrainingSteps,
)
from rl_coach.environments.gym_environment import GymVectorEnvironment, mujoco_v2
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.batch_rl_graph_manager import BatchRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.episodic import EpisodicExperienceReplayParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule
from rl_coach.spaces import (
    DiscreteActionSpace,
    RewardSpace,
    SpacesDefinition,
    StateSpace,
    VectorObservationSpace,
)

# ####################
# # Graph Scheduling #
# ####################

schedule_params = ScheduleParameters()
# 50 epochs (we run train over all the dataset, every epoch) of training
schedule_params.improve_steps = TrainingSteps(50)
# we evaluate the model every epoch
schedule_params.steps_between_evaluation_periods = TrainingSteps(1)


#########
# Agent #
#########
# note that we have moved to BCQ, which will help the training to converge better and faster
agent_params = DDQNBCQAgentParameters()
agent_params.network_wrappers["main"].batch_size = 128
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = TrainingSteps(50)
agent_params.algorithm.discount = 0.99

# NN configuration
agent_params.network_wrappers["main"].learning_rate = 0.0001
agent_params.network_wrappers["main"].replace_mse_with_huber_loss = False

# ER - we'll be needing an episodic replay buffer for off-policy evaluation
agent_params.memory = EpisodicExperienceReplayParameters()

# E-Greedy schedule - there is no exploration in Batch RL. Disabling E-Greedy.
agent_params.exploration.epsilon_schedule = LinearSchedule(
    initial_value=0, final_value=0, decay_steps=1
)
agent_params.exploration.evaluation_epsilon = 0

# can use either a kNN or a NN based model for predicting which actions not to max over in the bellman equation
agent_params.algorithm.action_drop_method_parameters = KNNParameters()


#################
# Visualization #
#################

vis_params = VisualizationParameters()
vis_params.dump_gifs = True

################
#  Environment #
################
env_params = GymVectorEnvironment(level="CartPole-v0")

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = 150
preset_validation_params.max_episodes_to_achieve_reward = 250


graph_manager = BatchRLGraphManager(
    agent_params=agent_params,
    env_params=env_params,
    schedule_params=schedule_params,
    vis_params=vis_params,
    preset_validation_params=preset_validation_params,
)
