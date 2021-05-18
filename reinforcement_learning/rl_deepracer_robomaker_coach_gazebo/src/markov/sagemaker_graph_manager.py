import json

from markov.architecture.constants import Input
from markov.architecture.embedder_factory import create_input_embedder, create_middle_embedder
from markov.boto.s3.constants import TrainingAlgorithm
from markov.constants import ExplorationTypes, HyperParameterKeys, LossTypes
from markov.environments.deepracer_racetrack_env import DeepRacerRacetrackEnvParameters
from markov.exploration_policies.deepracer_categorical import DeepRacerCategoricalParameters
from markov.filters.observation.observation_binary_filter import ObservationBinarySectorFilter
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.memories.deepracer_memory import DeepRacerMemoryParameters
from markov.multi_agent_coach.agents.sac_agent import SoftActorCriticAgentParameters
from markov.multi_agent_coach.multi_agent_graph_manager import MultiAgentGraphManager
from markov.multi_agent_coach.spaces import ScalableBoxActionSpace
from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.base_parameters import (
    DistributedCoachSynchronizationType,
    PresetValidationParameters,
    RunType,
    VisualizationParameters,
)
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.exploration_policies.additive_noise import AdditiveNoiseParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_clipping_filter import ObservationClippingFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.non_episodic.experience_replay import ExperienceReplayParameters
from rl_coach.schedules import LinearSchedule
from rl_coach.spaces import DiscreteActionSpace


class DeepRacerClippedPPOAgentParams(ClippedPPOAgentParameters):
    def __init__(self):
        super().__init__()
        # Agent to pass to the enviroment class, the objects in this list should
        # adhere to the AgentInterface
        self.env_agent = None


class DeepRacerSACAgentParams(SoftActorCriticAgentParameters):
    def __init__(self):
        super().__init__()
        # Agent to pass to the enviroment class, the objects in this list should
        # adhere to the AgentInterface
        self.env_agent = None


def get_sac_params(agent_params, agent, params, run_type=str(RunType.ROLLOUT_WORKER)):
    for net_key in ["policy", "v", "q"]:
        agent_params.network_wrappers[net_key].learning_rate = params[
            HyperParameterKeys.LEARNING_RATE.value
        ]
        agent_params.network_wrappers[net_key].input_embedders_parameters = create_input_embedder(
            agent.network_settings["input_embedders"],
            agent.network_settings["embedder_type"],
            agent.network_settings["activation_function"],
        )
        # DH: use empty middleware_embedder for q net
        if net_key != "q":
            agent_params.network_wrappers[net_key].middleware_parameters = create_middle_embedder(
                agent.network_settings["middleware_embedders"],
                agent.network_settings["embedder_type"],
                agent.network_settings["activation_function"],
            )

        for net_key in ["policy", "q", "v"]:
            agent_params.network_wrappers[net_key].batch_size = params[
                HyperParameterKeys.BATCH_SIZE.value
            ]
            agent_params.network_wrappers[net_key].optimizer_epsilon = 1e-5
            agent_params.network_wrappers[net_key].adam_optimizer_beta2 = 0.999
            if params[HyperParameterKeys.LOSS_TYPE.value] == LossTypes.HUBER.value:
                agent_params.network_wrappers[net_key].replace_mse_with_huber_loss = True
    agent_params.network_wrappers["policy"].heads_parameters[0].sac_alpha = params[
        HyperParameterKeys.SAC_ALPHA.value
    ]
    # Rescale action values in the policy head
    agent_params.network_wrappers["policy"].heads_parameters[0].rescale_action_values = True
    agent_params.algorithm.discount = params[HyperParameterKeys.DISCOUNT_FACTOR.value]
    # DH: should set num_steps_between_copying_online_weights_to_target as EnvironmentSteps instead of EnvironmentEpisodes.
    # see agent.py should_copy_online_weight...
    agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(
        params[HyperParameterKeys.NUM_EPISODES_BETWEEN_TRAINING.value]
    )
    agent_params.algorithm.distributed_coach_synchronization_type = (
        DistributedCoachSynchronizationType.SYNC
    )
    # tau=1
    agent_params.algorithm.rate_for_copying_weights_to_target = 1
    agent_params.algorithm.use_deterministic_for_evaluation = True

    # DH: ----to address the training worker fetch issue--------------------------
    if run_type == str(RunType.TRAINER):
        agent_params.memory = ExperienceReplayParameters()
    elif run_type == str(RunType.ROLLOUT_WORKER):
        agent_params.memory = DeepRacerMemoryParameters()  # EpisodicExperienceReplayParameters()
    return agent_params


def get_clipped_ppo_params(agent_params, agent, params):
    """This function is algorithm specific settings required for Clipped PPO algorithm

    Args:
        agent_params (DeepRacerClippedPPOAgentParams): the agent parameters that will be used to create the RL agent
        agent (Agent): The agent object that was created either as part of create_rollout_agent or create_training_agent
        params (dict): dictionary of hyperparameters

    Returns:
        DeepRacerClippedPPOAgentParams: updated agent params object with hyperparameters and other required details
    """
    agent_params.network_wrappers["main"].learning_rate = params[
        HyperParameterKeys.LEARNING_RATE.value
    ]

    agent_params.network_wrappers["main"].input_embedders_parameters = create_input_embedder(
        agent.network_settings["input_embedders"],
        agent.network_settings["embedder_type"],
        agent.network_settings["activation_function"],
    )
    agent_params.network_wrappers["main"].middleware_parameters = create_middle_embedder(
        agent.network_settings["middleware_embedders"],
        agent.network_settings["embedder_type"],
        agent.network_settings["activation_function"],
    )

    agent_params.network_wrappers["main"].batch_size = params[HyperParameterKeys.BATCH_SIZE.value]
    agent_params.network_wrappers["main"].optimizer_epsilon = 1e-5
    agent_params.network_wrappers["main"].adam_optimizer_beta2 = 0.999

    if params[HyperParameterKeys.LOSS_TYPE.value] == LossTypes.HUBER.value:
        agent_params.network_wrappers["main"].replace_mse_with_huber_loss = True

    agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
    agent_params.algorithm.beta_entropy = params[HyperParameterKeys.BETA_ENTROPY.value]
    agent_params.algorithm.gae_lambda = 0.95
    agent_params.algorithm.discount = params[HyperParameterKeys.DISCOUNT_FACTOR.value]
    agent_params.algorithm.optimization_epochs = params[HyperParameterKeys.NUM_EPOCHS.value]
    agent_params.algorithm.estimate_state_value_using_gae = True
    agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentEpisodes(
        params[HyperParameterKeys.NUM_EPISODES_BETWEEN_TRAINING.value]
    )
    agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(
        params[HyperParameterKeys.NUM_EPISODES_BETWEEN_TRAINING.value]
    )

    agent_params.algorithm.distributed_coach_synchronization_type = (
        DistributedCoachSynchronizationType.SYNC
    )
    if (
        params[HyperParameterKeys.EXPLORATION_TYPE.value].lower().strip()
        == ExplorationTypes.CATEGORICAL.value
    ):
        agent_params.exploration = {
            DiscreteActionSpace: DeepRacerCategoricalParameters(
                use_stochastic_evaluation_policy=False
            ),
            ScalableBoxActionSpace: AdditiveNoiseParameters(),
        }
    elif (
        params[HyperParameterKeys.EXPLORATION_TYPE.value].lower().strip()
        == ExplorationTypes.E_GREEDY.value
    ):
        agent_params.exploration = {
            DiscreteActionSpace: EGreedyParameters(),
            ScalableBoxActionSpace: AdditiveNoiseParameters(),
        }
        agent_params.exploration[DiscreteActionSpace].epsilon_schedule = LinearSchedule(
            1.0,
            params[HyperParameterKeys.E_GREEDY_VALUE.value],
            params[HyperParameterKeys.EPSILON_STEPS.value],
        )
    else:
        log_and_exit(
            "Unknown exploration_type found in hyper parameters. \
            exploration_type: {}".format(
                params[HyperParameterKeys.EXPLORATION_TYPE.value].lower().strip()
            ),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )

    agent_params.memory = DeepRacerMemoryParameters()
    return agent_params


# % TODO - refactor this module to be more modular based on the training algorithm and avoid if-else
def get_updated_hyper_parameters(hp_dict, training_algorithm):
    """Update the default hyperparameters

    Args:
        hp_dict (dict): Hyperparameters passed when training job is created
        training_algorithm (str): Training algorithm value from TrainingAlgorithm enum

    Returns:
        params (dict): updated hyperparameters
    """
    ####################
    # All Default Parameters #
    ####################
    params = {}
    params[HyperParameterKeys.BATCH_SIZE.value] = int(
        hp_dict.get(HyperParameterKeys.BATCH_SIZE.value, 64)
    )
    params[HyperParameterKeys.STACK_SIZE.value] = int(
        hp_dict.get(HyperParameterKeys.STACK_SIZE.value, 1)
    )
    params[HyperParameterKeys.LEARNING_RATE.value] = float(
        hp_dict.get(HyperParameterKeys.LEARNING_RATE.value, 0.0003)
    )
    params[HyperParameterKeys.EXPLORATION_TYPE.value] = (
        hp_dict.get(HyperParameterKeys.EXPLORATION_TYPE.value, ExplorationTypes.CATEGORICAL.value)
    ).lower()
    params[HyperParameterKeys.E_GREEDY_VALUE.value] = float(
        hp_dict.get(HyperParameterKeys.E_GREEDY_VALUE.value, 0.05)
    )
    params[HyperParameterKeys.EPSILON_STEPS.value] = int(
        hp_dict.get(HyperParameterKeys.EPSILON_STEPS.value, 10000)
    )
    params[HyperParameterKeys.DISCOUNT_FACTOR.value] = float(
        hp_dict.get(HyperParameterKeys.DISCOUNT_FACTOR.value, 0.999)
    )
    params[HyperParameterKeys.LOSS_TYPE.value] = hp_dict.get(
        HyperParameterKeys.LOSS_TYPE.value, LossTypes.MEAN_SQUARED_ERROR.value
    ).lower()
    params[HyperParameterKeys.NUM_EPISODES_BETWEEN_TRAINING.value] = int(
        hp_dict.get(HyperParameterKeys.NUM_EPISODES_BETWEEN_TRAINING.value, 20)
    )
    params[HyperParameterKeys.TERMINATION_CONDITION_MAX_EPISODES.value] = int(
        hp_dict.get(HyperParameterKeys.TERMINATION_CONDITION_MAX_EPISODES.value, 100000)
    )
    params[HyperParameterKeys.TERMINATION_CONDITION_AVG_SCORE.value] = float(
        hp_dict.get(HyperParameterKeys.TERMINATION_CONDITION_AVG_SCORE.value, 100000)
    )

    ####################
    # Clipped PPO algo
    ####################
    if TrainingAlgorithm.CLIPPED_PPO.value == training_algorithm:
        params[HyperParameterKeys.BETA_ENTROPY.value] = float(
            hp_dict.get(HyperParameterKeys.BETA_ENTROPY.value, 0.01)
        )
        params[HyperParameterKeys.NUM_EPOCHS.value] = int(
            hp_dict.get(HyperParameterKeys.NUM_EPOCHS.value, 10)
        )
    ####################
    # SAC algo
    ####################
    elif TrainingAlgorithm.SAC.value == training_algorithm:
        params[HyperParameterKeys.SAC_ALPHA.value] = float(
            hp_dict.get(HyperParameterKeys.SAC_ALPHA.value, 0.2)
        )
    return params


def get_graph_manager(
    hp_dict,
    agent_list,
    run_phase_subject,
    enable_domain_randomization=False,
    done_condition=any,
    run_type=str(RunType.ROLLOUT_WORKER),
    pause_physics=None,
    unpause_physics=None,
):
    ####################
    # Hyperparameters #
    ####################
    training_algorithm = (
        agent_list[0].ctrl.model_metadata.training_algorithm if agent_list else None
    )
    params = get_updated_hyper_parameters(hp_dict, training_algorithm)
    params_json = json.dumps(params, indent=2, sort_keys=True)
    print("Using the following hyper-parameters", params_json, sep="\n")

    ####################
    # Graph Scheduling #
    ####################
    schedule_params = ScheduleParameters()
    schedule_params.improve_steps = TrainingSteps(
        params[HyperParameterKeys.TERMINATION_CONDITION_MAX_EPISODES.value]
    )
    schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(40)
    schedule_params.evaluation_steps = EnvironmentEpisodes(5)
    schedule_params.heatup_steps = EnvironmentSteps(0)

    #########
    # Agent #
    #########
    trainable_agents_list = list()
    non_trainable_agents_list = list()

    for agent in agent_list:
        if agent.network_settings:
            if TrainingAlgorithm.SAC.value == training_algorithm:
                agent_params = get_sac_params(DeepRacerSACAgentParams(), agent, params, run_type)
            else:
                agent_params = get_clipped_ppo_params(
                    DeepRacerClippedPPOAgentParams(), agent, params
                )
            agent_params.env_agent = agent
            input_filter = InputFilter(is_a_reference_filter=True)
            for observation in agent.network_settings["input_embedders"].keys():
                if (
                    observation == Input.LEFT_CAMERA.value
                    or observation == Input.CAMERA.value
                    or observation == Input.OBSERVATION.value
                ):
                    input_filter.add_observation_filter(
                        observation, "to_grayscale", ObservationRGBToYFilter()
                    )
                    input_filter.add_observation_filter(
                        observation, "to_uint8", ObservationToUInt8Filter(0, 255)
                    )
                    input_filter.add_observation_filter(
                        observation, "stacking", ObservationStackingFilter(1)
                    )

                if observation == Input.STEREO.value:
                    input_filter.add_observation_filter(
                        observation, "to_uint8", ObservationToUInt8Filter(0, 255)
                    )

                if observation == Input.LIDAR.value:
                    input_filter.add_observation_filter(
                        observation, "clipping", ObservationClippingFilter(0.15, 1.0)
                    )
                if observation == Input.SECTOR_LIDAR.value:
                    input_filter.add_observation_filter(
                        observation, "binary", ObservationBinarySectorFilter()
                    )
            agent_params.input_filter = input_filter()
            trainable_agents_list.append(agent_params)
        else:
            non_trainable_agents_list.append(agent)

    ###############
    # Environment #
    ###############
    env_params = DeepRacerRacetrackEnvParameters()
    env_params.agents_params = trainable_agents_list
    env_params.non_trainable_agents = non_trainable_agents_list
    env_params.level = "DeepRacerRacetrackEnv-v0"
    env_params.run_phase_subject = run_phase_subject
    env_params.enable_domain_randomization = enable_domain_randomization
    env_params.done_condition = done_condition
    env_params.pause_physics = pause_physics
    env_params.unpause_physics = unpause_physics
    vis_params = VisualizationParameters()
    vis_params.dump_mp4 = False

    ########
    # Test #
    ########
    preset_validation_params = PresetValidationParameters()
    preset_validation_params.test = True
    preset_validation_params.min_reward_threshold = 400
    preset_validation_params.max_episodes_to_achieve_reward = 10000

    graph_manager = MultiAgentGraphManager(
        agents_params=trainable_agents_list,
        env_params=env_params,
        schedule_params=schedule_params,
        vis_params=vis_params,
        preset_validation_params=preset_validation_params,
        done_condition=done_condition,
    )
    return graph_manager, params_json
