import copy
import glob
import os
import time
from collections import OrderedDict
from distutils.dir_util import copy_tree, remove_tree
from typing import Dict, List, Tuple
import contextlib

from rl_coach.base_parameters import iterable_to_items, TaskParameters, DistributedTaskParameters, Frameworks, \
    VisualizationParameters, PresetValidationParameters, RunType, AgentParameters
from rl_coach.checkpoint import CheckpointStateUpdater, get_checkpoint_state, SingleCheckpoint, CheckpointState
from rl_coach.core_types import TotalStepsCounter, RunPhase, PlayingStepsType, TrainingSteps, EnvironmentEpisodes, \
    EnvironmentSteps, StepMethod, Transition, TimeTypes
from rl_coach.environments.environment import Environment, EnvironmentParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.logger import screen, Logger
from rl_coach.saver import SaverCollection
from rl_coach.utils import set_cpu, short_dynamic_import
from rl_coach.data_stores.data_store_impl import get_data_store as data_store_creator
from rl_coach.memories.backend.memory_impl import get_memory_backend
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.checkpoint import CheckpointStateReader

import markov.deepracer_memory_multi as deepracer_memory
from markov.multi_agent_coach.multi_agent_level_manager import MultiAgentLevelManager


class MultiAgentGraphManager(object):
    """
    A simple multi-agent graph manager and a single environment which is interacted with.
    """
    def __init__(self, agents_params: List[AgentParameters], env_params: EnvironmentParameters,
                 schedule_params: ScheduleParameters,
                 vis_params: VisualizationParameters = VisualizationParameters(),
                 preset_validation_params: PresetValidationParameters = PresetValidationParameters()):
        self.sess = {agent_params.name: None for agent_params in agents_params}
        self.level_managers = []  # type: List[MultiAgentLevelManager]
        self.top_level_manager = None
        self.environments = []
        self.set_schedule_params(schedule_params)
        self.visualization_parameters = vis_params
        self.name = 'multi_agent_graph'
        self.task_parameters = None
        self._phase = self.phase = RunPhase.UNDEFINED
        self.preset_validation_params = preset_validation_params
        self.reset_required = False
        self.num_checkpoints_to_keep = 4 # TODO: make this a parameter

        # timers
        self.graph_creation_time = None
        self.last_checkpoint_saving_time = time.time()

        # counters
        self.total_steps_counters = {
            RunPhase.HEATUP: TotalStepsCounter(),
            RunPhase.TRAIN: TotalStepsCounter(),
            RunPhase.TEST: TotalStepsCounter()
        }
        self.checkpoint_id = 0

        self.checkpoint_saver = {agent_params.name: None for agent_params in agents_params}
        self.checkpoint_state_updater = None
        self.graph_logger = Logger()
        self.data_store = None
        self.is_batch_rl = False
        self.time_metric = TimeTypes.EpisodeNumber

        self.env_params = env_params
        self.agents_params = agents_params
        self.agent_params = agents_params[0] # ...(find a better way)...

        for agent_index, agent_params in enumerate(agents_params):
            if len(agents_params) == 1:
                agent_params.name = "agent"
            else:
                agent_params.name = "agent_{}".format(agent_index)
            agent_params.visualization = copy.copy(vis_params)
            if agent_params.input_filter is None:
                agent_params.input_filter = copy.copy(env_params.default_input_filter())
            if agent_params.output_filter is None:
                agent_params.output_filter = copy.copy(env_params.default_output_filter())

    def create_graph(self, task_parameters=TaskParameters(),
                     stop_physics=None, start_physics=None, empty_service_call=None):
        self.graph_creation_time = time.time()
        self.task_parameters = task_parameters

        if isinstance(task_parameters, DistributedTaskParameters):
            screen.log_title("Creating graph - name: {} task id: {} type: {}".format(self.__class__.__name__,
                                                                                     task_parameters.task_index,
                                                                                     task_parameters.job_type))
        else:
            screen.log_title("Creating graph - name: {}".format(self.__class__.__name__))

        # "hide" the gpu if necessary
        if task_parameters.use_cpu:
            set_cpu()

        # create a target server for the worker and a device
        if isinstance(task_parameters, DistributedTaskParameters):
            task_parameters.worker_target, task_parameters.device = \
                self.create_worker_or_parameters_server(task_parameters=task_parameters)
        # If necessary start the physics and then stop it after agent creation
        if start_physics and empty_service_call:
            start_physics(empty_service_call())
        # create the graph modules
        self.level_managers, self.environments = self._create_graph(task_parameters)
        if stop_physics and empty_service_call:
            stop_physics(empty_service_call())
        # set self as the parent of all the level managers
        self.top_level_manager = self.level_managers[0]
        for level_manager in self.level_managers:
            level_manager.parent_graph_manager = self

        # create a session (it needs to be created after all the graph ops were created)
        self.sess = {agent_params.name: None for agent_params in self.agents_params}
        self.create_session(task_parameters=task_parameters)

        self._phase = self.phase = RunPhase.UNDEFINED

        self.setup_logger()

        return self

    def _create_graph(self, task_parameters: TaskParameters) -> Tuple[List[MultiAgentLevelManager], List[Environment]]:
        # environment loading
        self.env_params.seed = task_parameters.seed
        self.env_params.experiment_path = task_parameters.experiment_path
        env = short_dynamic_import(self.env_params.path)(**self.env_params.__dict__,
                                                         visualization_parameters=self.visualization_parameters)

        # agent loading
        agents = OrderedDict()
        for agent_params in self.agents_params:
            agent_params.task_parameters = copy.copy(task_parameters)
            agent = short_dynamic_import(agent_params.path)(agent_params)
            agents[agent_params.name] = agent

            if hasattr(self, 'memory_backend_params') and \
                    self.memory_backend_params.run_type == str(RunType.ROLLOUT_WORKER):
                agent.memory.memory_backend = deepracer_memory.DeepRacerRolloutBackEnd(self.memory_backend_params,
                                                                                       agent_params.algorithm.num_consecutive_playing_steps,
                                                                                       agent_params.name)

        # set level manager
        level_manager = MultiAgentLevelManager(agents=agents, environment=env, name="main_level")

        return [level_manager], [env]

    @staticmethod
    def _create_worker_or_parameters_server_tf(task_parameters: DistributedTaskParameters):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
        config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1

        from rl_coach.architectures.tensorflow_components.distributed_tf_utils import \
            create_and_start_parameters_server, \
            create_cluster_spec, create_worker_server_and_device

        # create cluster spec
        cluster_spec = create_cluster_spec(parameters_server=task_parameters.parameters_server_hosts,
                                           workers=task_parameters.worker_hosts)

        # create and start parameters server (non-returning function) or create a worker and a device setter
        if task_parameters.job_type == "ps":
            create_and_start_parameters_server(cluster_spec=cluster_spec,
                                               config=config)
        elif task_parameters.job_type == "worker":
            return create_worker_server_and_device(cluster_spec=cluster_spec,
                                                   task_index=task_parameters.task_index,
                                                   use_cpu=task_parameters.use_cpu,
                                                   config=config)
        else:
            raise ValueError("The job type should be either ps or worker and not {}"
                             .format(task_parameters.job_type))

    @staticmethod
    def create_worker_or_parameters_server(task_parameters: DistributedTaskParameters):
        if task_parameters.framework_type == Frameworks.tensorflow:
            return GraphManager._create_worker_or_parameters_server_tf(task_parameters)
        elif task_parameters.framework_type == Frameworks.mxnet:
            raise NotImplementedError('Distributed training not implemented for MXNet')
        else:
            raise ValueError('Invalid framework {}'.format(task_parameters.framework_type))

    def _create_session_tf(self, task_parameters: TaskParameters):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.allow_soft_placement = True  # allow placing ops on cpu if they are not fit for gpu
        config.gpu_options.allow_growth = True  # allow the gpu memory allocated for the worker to grow if needed
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1

        if isinstance(task_parameters, DistributedTaskParameters):
            # the distributed tensorflow setting
            from rl_coach.architectures.tensorflow_components.distributed_tf_utils import create_monitored_session
            if hasattr(self.task_parameters, 'checkpoint_restore_path') and self.task_parameters.checkpoint_restore_path:
                checkpoint_dir = os.path.join(task_parameters.experiment_path, 'checkpoint')
                if os.path.exists(checkpoint_dir):
                    remove_tree(checkpoint_dir)
                # in the locally distributed case, checkpoints are always restored from a directory (and not from a
                # file)
                copy_tree(task_parameters.checkpoint_restore_path, checkpoint_dir)
            else:
                checkpoint_dir = task_parameters.checkpoint_save_dir

            self.sess = create_monitored_session(target=task_parameters.worker_target,
                                                 task_index=task_parameters.task_index,
                                                 checkpoint_dir=checkpoint_dir,
                                                 checkpoint_save_secs=task_parameters.checkpoint_save_secs,
                                                 config=config)
            # set the session for all the modules
            self.set_session(self.sess)
        else:
            # regular session
            self.sess = {agent_params.name: tf.Session(config=config) for agent_params in self.agents_params}
            # set the session for all the modules
            self.set_session(self.sess)

        # the TF graph is static, and therefore is saved once - in the beginning of the experiment
        if hasattr(self.task_parameters, 'checkpoint_save_dir') and self.task_parameters.checkpoint_save_dir:
            self.save_graph()

    def _create_session_mx(self):
        """
        Call set_session to initialize parameters and construct checkpoint_saver
        """
        self.set_session(sess=None)  # Initialize all modules

    def create_session(self, task_parameters: TaskParameters):
        if task_parameters.framework_type == Frameworks.tensorflow:
            self._create_session_tf(task_parameters)
        elif task_parameters.framework_type == Frameworks.mxnet:
            self._create_session_mx()
        else:
            raise ValueError('Invalid framework {}'.format(task_parameters.framework_type))

        # Create parameter saver
        self.checkpoint_saver = {agent_params.name: SaverCollection() for agent_params in self.agents_params}
        for level in self.level_managers:
            for agent_params in self.agents_params:
                self.checkpoint_saver[agent_params.name].update(level.collect_savers(agent_params.name))

        # restore from checkpoint if given
        self.restore_checkpoint()

    def save_graph(self) -> None:
        """
        Save the TF graph to a protobuf description file in the experiment directory
        :return: None
        """
        import tensorflow as tf

        # write graph
        tf.train.write_graph(tf.get_default_graph(),
                             logdir=self.task_parameters.checkpoint_save_dir,
                             name='graphdef.pb',
                             as_text=False)

    def _save_onnx_graph_tf(self) -> None:
        """
        Save the tensorflow graph as an ONNX graph.
        This requires the graph and the weights checkpoint to be stored in the experiment directory.
        It then freezes the graph (merging the graph and weights checkpoint), and converts it to ONNX.
        :return: None
        """
        # collect input and output nodes
        input_nodes = []
        output_nodes = []
        for level in self.level_managers:
            for agent in level.agents.values():
                for network in agent.networks.values():
                    for input_key, input in network.online_network.inputs.items():
                        if not input_key.startswith("output_"):
                            input_nodes.append(input.name)
                    for output in network.online_network.outputs:
                        output_nodes.append(output.name)

        from rl_coach.architectures.tensorflow_components.architecture import save_onnx_graph

        save_onnx_graph(input_nodes, output_nodes, self.task_parameters.checkpoint_save_dir)

    def save_onnx_graph(self) -> None:
        """
        Save the graph as an ONNX graph.
        This requires the graph and the weights checkpoint to be stored in the experiment directory.
        It then freezes the graph (merging the graph and weights checkpoint), and converts it to ONNX.
        :return: None
        """
        if self.task_parameters.framework_type == Frameworks.tensorflow:
            self._save_onnx_graph_tf()

    def setup_logger(self) -> None:
        # dump documentation
        logger_prefix = "{graph_name}".format(graph_name=self.name)
        self.graph_logger.set_logger_filenames(self.task_parameters.experiment_path, logger_prefix=logger_prefix,
                                               add_timestamp=True, task_id=self.task_parameters.task_index)
        if self.visualization_parameters.dump_parameters_documentation:
            self.graph_logger.dump_documentation(str(self))
        [manager.setup_logger() for manager in self.level_managers]

    @property
    def phase(self) -> RunPhase:
        """
        Get the phase of the graph
        :return: the current phase
        """
        return self._phase

    @phase.setter
    def phase(self, val: RunPhase):
        """
        Change the phase of the graph and all the hierarchy levels below it
        :param val: the new phase
        :return: None
        """
        self._phase = val
        for level_manager in self.level_managers:
            level_manager.phase = val
        for environment in self.environments:
            environment.phase = val

    @property
    def current_step_counter(self) -> TotalStepsCounter:
        return self.total_steps_counters[self.phase]

    @contextlib.contextmanager
    def phase_context(self, phase):
        """
        Create a context which temporarily sets the phase to the provided phase.
        The previous phase is restored afterwards.
        """
        old_phase = self.phase
        self.phase = phase
        yield
        self.phase = old_phase

    def set_session(self, sess) -> None:
        """
        Set the deep learning framework session for all the modules in the graph
        :return: None
        """
        [manager.set_session(sess) for manager in self.level_managers]

    def heatup(self, steps: PlayingStepsType) -> None:
        """
        Perform heatup for several steps, which means taking random actions and storing the results in memory
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        self.verify_graph_was_created()

        if steps.num_steps > 0:
            with self.phase_context(RunPhase.HEATUP):
                screen.log_title("{}: Starting heatup".format(self.name))

                # reset all the levels before starting to heatup
                self.reset_internal_state(force_environment_reset=True)

                # act for at least steps, though don't interrupt an episode
                count_end = self.current_step_counter + steps
                while self.current_step_counter < count_end:
                    self.act(EnvironmentEpisodes(1))

    def handle_episode_ended(self) -> None:
        """
        End an episode and reset all the episodic parameters
        :return: None
        """
        self.current_step_counter[EnvironmentEpisodes] += 1

        [environment.handle_episode_ended() for environment in self.environments]

    def train(self) -> None:
        """
        Perform several training iterations for all the levels in the hierarchy
        :param steps: number of training iterations to perform
        :return: None
        """
        self.verify_graph_was_created()

        with self.phase_context(RunPhase.TRAIN):
            self.current_step_counter[TrainingSteps] += 1
            [manager.train() for manager in self.level_managers]

    def reset_internal_state(self, force_environment_reset=False) -> None:
        """
        Reset an episode for all the levels
        :param force_environment_reset: force the environment to reset the episode even if it has some conditions that
                                        tell it not to. for example, if ale life is lost, gym will tell the agent that
                                        the episode is finished but won't actually reset the episode if there are more
                                        lives available
        :return: None
        """
        self.verify_graph_was_created()

        self.reset_required = False
        [environment.reset_internal_state(force_environment_reset) for environment in self.environments]
        [manager.reset_internal_state() for manager in self.level_managers]

    def act(self, steps: PlayingStepsType, wait_for_full_episodes=False) -> None:
        """
        Do several steps of acting on the environment
        :param wait_for_full_episodes: if set, act for at least `steps`, but make sure that the last episode is complete
        :param steps: the number of steps as a tuple of steps time and steps count
        """
        self.verify_graph_was_created()

        # perform several steps of playing
        count_end = self.current_step_counter + steps
        done = False
        while self.current_step_counter < count_end or (wait_for_full_episodes and not done):
            # reset the environment if the previous episode was terminated
            if self.reset_required:
                self.reset_internal_state()

            steps_begin = self.environments[0].total_steps_counter
            done = self.top_level_manager.step(None)
            steps_end = self.environments[0].total_steps_counter

            if done:
                self.handle_episode_ended()
                self.reset_required = True

            self.current_step_counter[EnvironmentSteps] += (steps_end - steps_begin)

            # if no steps were made (can happen when no actions are taken while in the TRAIN phase, either in batch RL
            # or in imitation learning), we force end the loop, so that it will not continue forever.
            if (steps_end - steps_begin) == 0:
                break

    def train_and_act(self, steps: StepMethod) -> None:
        """
        Train the agent by doing several acting steps followed by several training steps continually
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: None
        """
        self.verify_graph_was_created()

        # perform several steps of training interleaved with acting
        if steps.num_steps > 0:
            with self.phase_context(RunPhase.TRAIN):
                self.reset_internal_state(force_environment_reset=True)

                count_end = self.current_step_counter + steps
                while self.current_step_counter < count_end:
                    # The actual number of steps being done on the environment
                    # is decided by the agent, though this inner loop always
                    # takes at least one step in the environment (at the GraphManager level).
                    # The agent might also decide to skip acting altogether.
                    # Depending on internal counters and parameters, it doesn't always train or save checkpoints.
                    self.act(EnvironmentSteps(1))
                    self.train()
                    self.occasionally_save_checkpoint()

    def sync(self) -> None:
        """
        Sync the global network parameters to the graph
        :return:
        """
        [manager.sync() for manager in self.level_managers]

    def evaluate(self, steps: PlayingStepsType) -> bool:
        """
        Perform evaluation for several steps
        :param steps: the number of steps as a tuple of steps time and steps count
        :return: bool, True if the target reward and target success has been reached
        """
        self.verify_graph_was_created()

        if steps.num_steps > 0:
            with self.phase_context(RunPhase.TEST):
                # reset all the levels before starting to evaluate
                self.reset_internal_state(force_environment_reset=True)
                self.sync()

                # act for at least `steps`, though don't interrupt an episode
                count_end = self.current_step_counter + steps
                while self.current_step_counter < count_end:
                    self.act(EnvironmentEpisodes(1))
                    self.sync()
        if self.should_stop():
            self.flush_finished()
            screen.success("Reached required success rate. Exiting.")
            return True
        return False

    def improve(self):
        """
        The main loop of the run.
        Defined in the following steps:
        1. Heatup
        2. Repeat:
            2.1. Repeat:
                2.1.1. Act
                2.1.2. Train
                2.1.3. Possibly save checkpoint
            2.2. Evaluate
        :return: None
        """

        self.verify_graph_was_created()

        # initialize the network parameters from the global network
        self.sync()

        # heatup
        self.heatup(self.heatup_steps)

        # improve
        if self.task_parameters.task_index is not None:
            screen.log_title("Starting to improve {} task index {}".format(self.name, self.task_parameters.task_index))
        else:
            screen.log_title("Starting to improve {}".format(self.name))

        count_end = self.total_steps_counters[RunPhase.TRAIN] + self.improve_steps
        while self.total_steps_counters[RunPhase.TRAIN] < count_end:
            self.train_and_act(self.steps_between_evaluation_periods)
            if self.evaluate(self.evaluation_steps):
                break

    def restore_checkpoint(self):
        self.verify_graph_was_created()

        # TODO: find better way to load checkpoints that were saved with a global network into the online network
        if self.task_parameters.checkpoint_restore_path:
            restored_checkpoint_paths = []
            for agent_params in self.agents_params:
                if len(self.agents_params) == 1:
                    agent_checkpoint_restore_path = self.task_parameters.checkpoint_restore_path
                else:
                    agent_checkpoint_restore_path = os.path.join(self.task_parameters.checkpoint_restore_path, agent_params.name)
                if os.path.isdir(agent_checkpoint_restore_path):
                    # a checkpoint dir
                    if self.task_parameters.framework_type == Frameworks.tensorflow and\
                            'checkpoint' in os.listdir(agent_checkpoint_restore_path):
                        # TODO-fixme checkpointing
                        # MonitoredTrainingSession manages save/restore checkpoints autonomously. Doing so,
                        # it creates it own names for the saved checkpoints, which do not match the "{}_Step-{}.ckpt"
                        # filename pattern. The names used are maintained in a CheckpointState protobuf file named
                        # 'checkpoint'. Using Coach's '.coach_checkpoint' protobuf file, results in an error when trying to
                        # restore the model, as the checkpoint names defined do not match the actual checkpoint names.
                        raise NotImplementedError('Checkpointing not implemented for TF monitored training session')
                    else:
                        checkpoint = get_checkpoint_state(agent_checkpoint_restore_path, all_checkpoints=True)

                    if checkpoint is None:
                        raise ValueError("No checkpoint to restore in: {}".format(agent_checkpoint_restore_path))
                    model_checkpoint_path = checkpoint.model_checkpoint_path
                    checkpoint_restore_dir = self.task_parameters.checkpoint_restore_path
                    restored_checkpoint_paths.append(model_checkpoint_path)

                    # Set the last checkpoint ID - only in the case of the path being a dir
                    chkpt_state_reader = CheckpointStateReader(self.task_parameters.checkpoint_restore_path,
                                                               checkpoint_state_optional=False)
                    self.checkpoint_id = chkpt_state_reader.get_latest().num + 1
                else:
                    # a checkpoint file
                    if self.task_parameters.framework_type == Frameworks.tensorflow:
                        model_checkpoint_path = self.task_parameters.checkpoint_restore_path
                        checkpoint_restore_dir = os.path.dirname(model_checkpoint_path)
                        restored_checkpoint_paths.append(model_checkpoint_path)
                    else:
                        raise ValueError("Currently restoring a checkpoint using the --checkpoint_restore_file argument is"
                                         " only supported when with tensorflow.")

                try:
                    self.checkpoint_saver[agent_params.name].restore(self.sess[agent_params.name],
                                                                     model_checkpoint_path)
                except Exception as ex:
                    raise ValueError("Failed to restore {}'s checkpoint: {}".format(agent_params.name, ex))

                all_checkpoints = sorted(list(set([c.name for c in checkpoint.all_checkpoints]))) # remove duplicates :-(
                if self.num_checkpoints_to_keep < len(all_checkpoints):
                    checkpoint_to_delete = all_checkpoints[-self.num_checkpoints_to_keep - 1]
                    agent_checkpoint_to_delete = os.path.join(agent_checkpoint_restore_path, checkpoint_to_delete)
                    for file in glob.glob("{}*".format(agent_checkpoint_to_delete)):
                        os.remove(file)

            [manager.restore_checkpoint(checkpoint_restore_dir) for manager in self.level_managers]
            [manager.post_training_commands() for manager in self.level_managers]

            screen.log_dict(
                OrderedDict([
                    ("Restoring from path", restored_checkpoint_path) for restored_checkpoint_path in restored_checkpoint_paths
                ]),
                prefix="Checkpoint"
            )

    def _get_checkpoint_state_tf(self, checkpoint_restore_dir):
        import tensorflow as tf
        return tf.train.get_checkpoint_state(checkpoint_restore_dir)

    def occasionally_save_checkpoint(self):
        # only the chief process saves checkpoints
        if self.task_parameters.checkpoint_save_secs \
                and time.time() - self.last_checkpoint_saving_time >= self.task_parameters.checkpoint_save_secs \
                and (self.task_parameters.task_index == 0  # distributed
                     or self.task_parameters.task_index is None  # single-worker
                     ):
            self.save_checkpoint()

    def save_checkpoint(self):
        # create current session's checkpoint directory
        if self.task_parameters.checkpoint_save_dir is None:
            self.task_parameters.checkpoint_save_dir = os.path.join(self.task_parameters.experiment_path, 'checkpoint')

        if not os.path.exists(self.task_parameters.checkpoint_save_dir):
            os.mkdir(self.task_parameters.checkpoint_save_dir)  # Create directory structure

        if self.checkpoint_state_updater is None:
            self.checkpoint_state_updater = CheckpointStateUpdater(self.task_parameters.checkpoint_save_dir)

        checkpoint_name = "{}_Step-{}.ckpt".format(
            self.checkpoint_id, self.total_steps_counters[RunPhase.TRAIN][EnvironmentSteps])

        saved_checkpoint_paths = []
        for agent_params in self.agents_params:
            if len(self.agents_params) == 1:
                agent_checkpoint_save_dir =self.task_parameters.checkpoint_save_dir
            else:
                agent_checkpoint_save_dir = os.path.join(self.task_parameters.checkpoint_save_dir, agent_params.name)
            if not os.path.exists(agent_checkpoint_save_dir):
                os.mkdir(agent_checkpoint_save_dir)

            agent_checkpoint_path = os.path.join(agent_checkpoint_save_dir, checkpoint_name)
            if not isinstance(self.task_parameters, DistributedTaskParameters):
                saved_checkpoint_paths.append(self.checkpoint_saver[agent_params.name].save(self.sess[agent_params.name], agent_checkpoint_path))
            else:
                saved_checkpoint_paths.append(agent_checkpoint_path)

            if self.num_checkpoints_to_keep < len(self.checkpoint_state_updater.all_checkpoints):
                checkpoint_to_delete = self.checkpoint_state_updater.all_checkpoints[-self.num_checkpoints_to_keep - 1]
                agent_checkpoint_to_delete = os.path.join(agent_checkpoint_save_dir, checkpoint_to_delete.name)
                for file in glob.glob("{}*".format(agent_checkpoint_to_delete)):
                    os.remove(file)

        # this is required in order for agents to save additional information like a DND for example
        [manager.save_checkpoint(checkpoint_name) for manager in self.level_managers]

        # Purge Redis memory after saving the checkpoint as Transitions are no longer needed at this point.
        if hasattr(self, 'memory_backend'):
            self.memory_backend.memory_purge()

        # the ONNX graph will be stored only if checkpoints are stored and the -onnx flag is used
        if self.task_parameters.export_onnx_graph:
            self.save_onnx_graph()

        # write the new checkpoint name to a file to signal this checkpoint has been fully saved
        self.checkpoint_state_updater.update(SingleCheckpoint(self.checkpoint_id, checkpoint_name))

        screen.log_dict(
            OrderedDict([
                ("Saving in path", saved_checkpoint_path) for saved_checkpoint_path in saved_checkpoint_paths
            ]),
            prefix="Checkpoint"
        )

        self.checkpoint_id += 1
        self.last_checkpoint_saving_time = time.time()

        if hasattr(self, 'data_store_params'):
            data_store = self.get_data_store(self.data_store_params)
            data_store.save_to_store()

    def verify_graph_was_created(self):
        """
        Verifies that the graph was already created, and if not, it creates it with the default task parameters
        :return: None
        """
        if self.graph_creation_time is None:
            self.create_graph()

    def __str__(self):
        result = ""
        for key, val in self.__dict__.items():
            params = ""
            if isinstance(val, list) or isinstance(val, dict) or isinstance(val, OrderedDict):
                items = iterable_to_items(val)
                for k, v in items:
                    params += "{}: {}\n".format(k, v)
            else:
                params = val
            result += "{}: \n{}\n".format(key, params)

        return result

    def should_train(self) -> bool:
        return any([manager.should_train() for manager in self.level_managers])

    # TODO-remove - this is a temporary flow, used by the trainer worker, duplicated from observe() - need to create
    #               an external trainer flow reusing the existing flow and methods [e.g. observe(), step(), act()]
    def emulate_act_on_trainer(self, steps: PlayingStepsType, transitions: Dict[str, Transition]) -> None:
        """
        This emulates the act using the transition obtained from the rollout worker on the training worker
        in case of distributed training.
        Do several steps of acting on the environment
        :param steps: the number of steps as a tuple of steps time and steps count
        """
        self.verify_graph_was_created()

        # perform several steps of playing
        count_end = self.current_step_counter + steps
        while self.current_step_counter < count_end:
            # reset the environment if the previous episode was terminated
            if self.reset_required:
                self.reset_internal_state()

            steps_begin = self.environments[0].total_steps_counter
            done = self.top_level_manager.emulate_step_on_trainer(transitions)
            steps_end = self.environments[0].total_steps_counter

            # add the diff between the total steps before and after stepping, such that environment initialization steps
            # (like in Atari) will not be counted.
            # We add at least one step so that even if no steps were made (in case no actions are taken in the training
            # phase), the loop will end eventually.
            self.current_step_counter[EnvironmentSteps] += max(1, steps_end - steps_begin)

            if done:
                self.handle_episode_ended()
                self.reset_required = True

    def fetch_from_worker(self, num_consecutive_playing_steps=None):
        if hasattr(self, 'memory_backend'):
            for transitions in self.memory_backend.fetch(num_consecutive_playing_steps):
                self.emulate_act_on_trainer(EnvironmentSteps(1), transitions)

    def setup_memory_backend(self) -> None:
        if hasattr(self, 'memory_backend_params'):
            self.memory_backend = deepracer_memory.DeepRacerTrainerBackEnd(self.memory_backend_params, self.agents_params)

    def should_stop(self) -> bool:
        return self.task_parameters.apply_stop_condition and all([manager.should_stop() for manager in self.level_managers])

    def get_data_store(self, param):
        if self.data_store:
            return self.data_store

        return data_store_creator(param)

    def signal_ready(self):
        if self.task_parameters.checkpoint_save_dir and os.path.exists(self.task_parameters.checkpoint_save_dir):
                open(os.path.join(self.task_parameters.checkpoint_save_dir, SyncFiles.TRAINER_READY.value), 'w').close()
        if hasattr(self, 'data_store_params'):
                data_store = self.get_data_store(self.data_store_params)
                data_store.save_to_store()

    def close(self) -> None:
        """
        Clean up to close environments.

        :return: None
        """
        for env in self.environments:
            env.close()

    def get_current_episodes_count(self):
        """
        Returns the current EnvironmentEpisodes counter
        """
        return self.current_step_counter[EnvironmentEpisodes]

    def flush_finished(self):
        """
        To indicate the training has finished, writes a `.finished` file to the checkpoint directory and calls
        the data store to updload that file.
        """
        if self.task_parameters.checkpoint_save_dir and os.path.exists(self.task_parameters.checkpoint_save_dir):
            open(os.path.join(self.task_parameters.checkpoint_save_dir, SyncFiles.FINISHED.value), 'w').close()
        if hasattr(self, 'data_store_params'):
            data_store = self.get_data_store(self.data_store_params)
            data_store.save_to_store()

    def set_schedule_params(self, schedule_params: ScheduleParameters):
        """
        Set schedule parameters for the graph.

        :param schedule_params: the schedule params to set.
        """
        self.heatup_steps = schedule_params.heatup_steps
        self.evaluation_steps = schedule_params.evaluation_steps
        self.steps_between_evaluation_periods = schedule_params.steps_between_evaluation_periods
        self.improve_steps = schedule_params.improve_steps
