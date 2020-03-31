import json

from markov.architecture.constants import Input
from markov.architecture.embedder_factory import create_input_embedder, create_middle_embedder
from markov.environments.deepracer_racetrack_env import DeepRacerRacetrackEnvParameters
from markov.multi_agent_coach.multi_agent_graph_manager import MultiAgentGraphManager

from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, \
                                     DistributedCoachSynchronizationType
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.exploration_policies.categorical import CategoricalParameters
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import InputFilter
from rl_coach.filters.observation.observation_rgb_to_y_filter import ObservationRGBToYFilter
from rl_coach.filters.observation.observation_stacking_filter import ObservationStackingFilter
from rl_coach.filters.observation.observation_to_uint8_filter import ObservationToUInt8Filter
from rl_coach.filters.observation.observation_clipping_filter import ObservationClippingFilter
from markov.filters.observation.observation_binary_filter import ObservationBinarySectorFilter
from markov.environments.constants import SECTOR_LIDAR_CLIPPING_DIST
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule


class DeepRacerAgentParams(ClippedPPOAgentParameters):
    def __init__(self):
        super().__init__()
        # Agent to pass to the enviroment class, the objects in this list should
        # adhere to the AgentInterface
        self.env_agent = None


def get_graph_manager(hp_dict, agent_list, run_phase_subject):
    ####################
    # All Default Parameters #
    ####################
    params = {}
    params["batch_size"] = int(hp_dict.get("batch_size", 64))
    params["num_epochs"] = int(hp_dict.get("num_epochs", 10))
    params["stack_size"] = int(hp_dict.get("stack_size", 1))
    params["lr"] = float(hp_dict.get("lr", 0.0003))
    params["exploration_type"] = (hp_dict.get("exploration_type", "categorical")).lower()
    params["e_greedy_value"] = float(hp_dict.get("e_greedy_value", .05))
    params["epsilon_steps"] = int(hp_dict.get("epsilon_steps", 10000))
    params["beta_entropy"] = float(hp_dict.get("beta_entropy", .01))
    params["discount_factor"] = float(hp_dict.get("discount_factor", .999))
    params["loss_type"] = hp_dict.get("loss_type", "Mean squared error").lower()
    params["num_episodes_between_training"] = int(hp_dict.get("num_episodes_between_training", 20))
    params["term_cond_max_episodes"] = int(hp_dict.get("term_cond_max_episodes", 100000))
    params["term_cond_avg_score"] = float(hp_dict.get("term_cond_avg_score", 100000))

    params_json = json.dumps(params, indent=2, sort_keys=True)
    print("Using the following hyper-parameters", params_json, sep='\n')

    ####################
    # Graph Scheduling #
    ####################
    schedule_params = ScheduleParameters()
    schedule_params.improve_steps = TrainingSteps(params["term_cond_max_episodes"])
    schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(40)
    schedule_params.evaluation_steps = EnvironmentEpisodes(5)
    schedule_params.heatup_steps = EnvironmentSteps(0)

    #########
    # Agent #
    #########
    trainable_agents_list = list()
    non_trainable_agents_list = list()

    for agent in agent_list:
        agent_params = DeepRacerAgentParams()
        if agent.network_settings:
            agent_params.env_agent = agent
            agent_params.network_wrappers['main'].learning_rate = params["lr"]

            agent_params.network_wrappers['main'].input_embedders_parameters = \
                create_input_embedder(agent.network_settings['input_embedders'],
                                      agent.network_settings['embedder_type'],
                                      agent.network_settings['activation_function'])
            agent_params.network_wrappers['main'].middleware_parameters = \
                create_middle_embedder(agent.network_settings['middleware_embedders'],
                                       agent.network_settings['embedder_type'],
                                       agent.network_settings['activation_function'])

            input_filter = InputFilter(is_a_reference_filter=True)
            for observation in agent.network_settings['input_embedders'].keys():
                if observation == Input.LEFT_CAMERA.value or observation == Input.CAMERA.value or\
                observation == Input.OBSERVATION.value:
                    input_filter.add_observation_filter(observation,
                                                        'to_grayscale', ObservationRGBToYFilter())
                    input_filter.add_observation_filter(observation,
                                                        'to_uint8',
                                                        ObservationToUInt8Filter(0, 255))
                    input_filter.add_observation_filter(observation,
                                                        'stacking', ObservationStackingFilter(1))

                if observation == Input.STEREO.value:
                    input_filter.add_observation_filter(observation,
                                                        'to_uint8',
                                                        ObservationToUInt8Filter(0, 255))

                if observation == Input.LIDAR.value:
                    input_filter.add_observation_filter(observation,
                                                        'clipping',
                                                        ObservationClippingFilter(0.15, 1.0))
                if observation == Input.SECTOR_LIDAR.value:
                    input_filter.add_observation_filter(observation,
                                                        'binary',
                                                        ObservationBinarySectorFilter())
            agent_params.input_filter = input_filter()

            agent_params.network_wrappers['main'].batch_size = params["batch_size"]
            agent_params.network_wrappers['main'].optimizer_epsilon = 1e-5
            agent_params.network_wrappers['main'].adam_optimizer_beta2 = 0.999

            if params["loss_type"] == "huber":
                agent_params.network_wrappers['main'].replace_mse_with_huber_loss = True

            agent_params.algorithm.clip_likelihood_ratio_using_epsilon = 0.2
            agent_params.algorithm.clipping_decay_schedule = LinearSchedule(1.0, 0, 1000000)
            agent_params.algorithm.beta_entropy = params["beta_entropy"]
            agent_params.algorithm.gae_lambda = 0.95
            agent_params.algorithm.discount = params["discount_factor"]
            agent_params.algorithm.optimization_epochs = params["num_epochs"]
            agent_params.algorithm.estimate_state_value_using_gae = True
            agent_params.algorithm.num_steps_between_copying_online_weights_to_target = \
                EnvironmentEpisodes(params["num_episodes_between_training"])
            agent_params.algorithm.num_consecutive_playing_steps = \
                EnvironmentEpisodes(params["num_episodes_between_training"])

            agent_params.algorithm.distributed_coach_synchronization_type = \
                DistributedCoachSynchronizationType.SYNC

            if params["exploration_type"] == "categorical":
                agent_params.exploration = CategoricalParameters()
            else:
                agent_params.exploration = EGreedyParameters()
                agent_params.exploration.epsilon_schedule = LinearSchedule(1.0,
                                                                           params["e_greedy_value"],
                                                                           params["epsilon_steps"])

            trainable_agents_list.append(agent_params)
        else:
            non_trainable_agents_list.append(agent)

    ###############
    # Environment #
    ###############
    env_params = DeepRacerRacetrackEnvParameters()
    env_params.agents_params = trainable_agents_list
    env_params.non_trainable_agents = non_trainable_agents_list
    env_params.level = 'DeepRacerRacetrackEnv-v0'
    env_params.run_phase_subject = run_phase_subject

    vis_params = VisualizationParameters()
    vis_params.dump_mp4 = False

    ########
    # Test #
    ########
    preset_validation_params = PresetValidationParameters()
    preset_validation_params.test = True
    preset_validation_params.min_reward_threshold = 400
    preset_validation_params.max_episodes_to_achieve_reward = 10000

    graph_manager = MultiAgentGraphManager(agents_params=trainable_agents_list,
                                           env_params=env_params,
                                           schedule_params=schedule_params, vis_params=vis_params,
                                           preset_validation_params=preset_validation_params)
    return graph_manager, params_json
