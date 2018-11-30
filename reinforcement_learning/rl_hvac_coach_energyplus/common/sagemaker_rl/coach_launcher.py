from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.agents.policy_gradients_agent import PolicyGradientsAgentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters, TaskParameters, Frameworks
from rl_coach.utils import short_dynamic_import
from rl_coach.core_types import SelectedPhaseOnlyDumpFilter, MaxDumpFilter, RunPhase
import rl_coach.core_types 
from rl_coach import logger
from rl_coach.logger import screen
import argparse
import copy
import logging
import os
import sys
import shutil
import glob
import re

from .configuration_list import ConfigurationList
from rl_coach.coach import CoachLauncher

screen.set_use_colors(False)  # Simple text logging so it looks good in CloudWatch

class CoachConfigurationList(ConfigurationList):
    """Helper Object for converting CLI arguments (or SageMaker hyperparameters) 
    into Coach configuration.
    """

    # Being security-paranoid and not instantiating any arbitrary string the customer passes in
    ALLOWED_TYPES = {
        'Frames': rl_coach.core_types.Frames,
        'EnvironmentSteps': rl_coach.core_types.EnvironmentSteps,
        'EnvironmentEpisodes': rl_coach.core_types.EnvironmentEpisodes,
        'TrainingSteps': rl_coach.core_types.TrainingSteps,
        'Time': rl_coach.core_types.Time,
    }



class SageMakerCoachPresetLauncher(CoachLauncher):
    """Base class for training RL tasks using RL-Coach.
    Customers subclass this to define specific kinds of workloads, overriding these methods as needed.
    """

    def __init__(self):
        super().__init__()
        self.hyperparams = None


    def get_config_args(self, parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Overrides the default CLI parsing.
        Sets the configuration parameters for what a SageMaker run should do.
        Note, this does not support the "play" mode.
        """
        # first, convert the parser to a Namespace object with all default values.
        empty_arg_list = []
        args, _ = parser.parse_known_args(args=empty_arg_list)
        parser = self.sagemaker_argparser()
        sage_args, unknown = parser.parse_known_args()
        
        # Now fill in the args that we care about.
        sagemaker_job_name = os.environ.get("sagemaker_job_name", "sagemaker-experiment")
        args.experiment_name = logger.get_experiment_name(sagemaker_job_name)
        
        # Override experiment_path used for outputs
        args.experiment_path = '/opt/ml/output/intermediate'
        rl_coach.logger.experiment_path = '/opt/ml/output/intermediate' # for gifs

        args.checkpoint_save_dir = '/opt/ml/output/data/checkpoint'
        args.checkpoint_save_secs = 10 # should avoid hardcoding
        # onnx for deployment for mxnet (not tensorflow)
        save_model = (sage_args.save_model == 1)
        backend = os.getenv('COACH_BACKEND', 'tensorflow')
        if save_model and backend == "mxnet":
            args.export_onnx_graph = True

        args.no_summary = True

        args.num_workers = sage_args.num_workers
        args.framework = Frameworks[backend]
        args.preset = sage_args.RLCOACH_PRESET
        # args.apply_stop_condition = True # uncomment for old coach behaviour

        self.hyperparameters = CoachConfigurationList()
        if len(unknown) % 2 == 1:
            raise ValueError("Odd number of command-line arguments specified. Key without value.")

        for i in range(0, len(unknown), 2):
            name = unknown[i]
            if name.startswith("--"):
                name = name[2:]
            else:
                raise ValueError("Unknown command-line argument %s" % name)
            val = unknown[i+1]
            self.map_hyperparameter(name, val)

        return args

    def map_hyperparameter(self, name, value):
        """This is a good method to override where customers can specify custom shortcuts
        for hyperparameters.  Default takes everything starting with "rl." and sends it
        straight to the graph manager.
        """
        if name.startswith("rl."):
            self.apply_hyperparameter(name, value)
        else:
            raise ValueError("Unknown hyperparameter %s" % name)


    def apply_hyperparameter(self, name, value):
        """Save this hyperparameter to be applied to the graph_manager object when
        it's ready.
        """
        print("Applying RL hyperparameter %s=%s" % (name,value))
        self.hyperparameters.store(name, value)


    def default_preset_name(self):
        """
        Sub-classes will typically return a single hard-coded string.
        """
        try:
            #TODO: remove this after converting all samples.
            default_preset = self.DEFAULT_PRESET
            screen.warning("Deprecated configuration of default preset.  Please implement default_preset_name()")
            return default_preset
        except:
            pass
        raise NotImplementedError("Sub-classes must specify the name of the default preset "+
                                  "for this RL problem.  This will be the name of a python "+
                                  "file (without .py) that defines a graph_manager variable")

    def sagemaker_argparser(self) -> argparse.ArgumentParser:
        """
        Expose only the CLI arguments that make sense in the SageMaker context.
        """
        parser = argparse.ArgumentParser()

        # Arguably this would be cleaner if we copied the config from the base class argparser.
        parser.add_argument('-n', '--num_workers',
                            help="(int) Number of workers for multi-process based agents, e.g. A3C",
                            default=1,
                            type=int)
        parser.add_argument('-p', '--RLCOACH_PRESET',
                            help="(string) Name of the file with the RLCoach preset",
                            default=self.default_preset_name(),
                            type=str)
        parser.add_argument('--save_model',
                            help="(int) Flag to save model artifact after training finish",
                            default=0,
                            type=int)
        return parser

    def path_of_main_launcher(self):
        """
        A bit of python magic to find the path of the file that launched the current process.
        """
        main_mod = sys.modules['__main__']
        try:
            launcher_file = os.path.abspath(sys.modules['__main__'].__file__)
            return os.path.dirname(launcher_file)
        except AttributeError:
            # If __main__.__file__ is missing, then we're probably in an interactive python shell
            return os.getcwd()

    def preset_from_name(self, preset_name):
        preset_path = self.path_of_main_launcher()
        print("Loading preset %s from %s" % (preset_name, preset_path))
        preset_path = os.path.join(self.path_of_main_launcher(),preset_name) + '.py:graph_manager'
        graph_manager = short_dynamic_import(preset_path, ignore_module_case=True)
        return graph_manager

    def get_graph_manager_from_args(self, args):
        # First get the graph manager for the customer-specified (or default) preset
        graph_manager = self.preset_from_name(args.preset)
        # Now override whatever config is specified in hyperparameters.
        self.hyperparameters.apply_subset(graph_manager, "rl.")
        # Set framework
        # Note: Some graph managers (e.g. HAC preset) create multiple agents and the attribute is called agents_params
        if hasattr(graph_manager, 'agent_params'):
            for network_parameters in graph_manager.agent_params.network_wrappers.values():
                network_parameters.framework = args.framework
        elif hasattr(graph_manager, 'agents_params'):
            for ap in graph_manager.agents_params:
                for network_parameters in ap.network_wrappers.values():
                    network_parameters.framework = args.framework
        return graph_manager

    def _save_tf_model(self):
        ckpt_dir = '/opt/ml/output/data/checkpoint'
        model_dir = '/opt/ml/model'

        # Re-Initialize from the checkpoint so that you will have the latest models up.
        tf.train.init_from_checkpoint(ckpt_dir,
                                      {'main_level/agent/online/network_0/': 'main_level/agent/online/network_0'})
        tf.train.init_from_checkpoint(ckpt_dir,
                                      {'main_level/agent/online/network_1/': 'main_level/agent/online/network_1'})

        # Create a new session with a new tf graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())  # initialize the checkpoint.

        # This is the node that will accept the input.
        input_nodes = tf.get_default_graph().get_tensor_by_name('main_level/agent/main/online/' + \
                                                                'network_0/observation/observation:0')
        # This is the node that will produce the output.
        output_nodes = tf.get_default_graph().get_operation_by_name('main_level/agent/main/online/' + \
                                                                    'network_1/ppo_head_0/policy')
        # Save the model as a servable model.
        tf.saved_model.simple_save(session=sess,
                                   export_dir='model',
                                   inputs={"observation": input_nodes},
                                   outputs={"policy": output_nodes.outputs[0]})
        # Move to the appropriate folder. Don't mind the directory, this just works.
        # rl-cart-pole is the name of the model. Remember it.
        shutil.move('model/', model_dir + '/model/tf-model/00000001/')
        # EASE will pick it up and upload to the right path.
        print("Success")

    def _save_onnx_model(self):
        from .onnx_utils import fix_onnx_model
        ckpt_dir = '/opt/ml/output/data/checkpoint'
        model_dir = '/opt/ml/model'
        # find latest onnx file
        # currently done by name, expected to be changed in future release of coach.
        glob_pattern = os.path.join(ckpt_dir, '*.onnx')
        onnx_files = [file for file in glob.iglob(glob_pattern, recursive=True)]
        if len(onnx_files) > 0:
            extract_step = lambda string: int(re.search('/(\d*)_Step.*', string, re.IGNORECASE).group(1))
            onnx_files.sort(key=extract_step)
            latest_onnx_file = onnx_files[-1]
            # move to model directory
            filepath_from = os.path.abspath(latest_onnx_file)
            filepath_to = os.path.join(model_dir, "model.onnx")
            shutil.move(filepath_from, filepath_to)
            fix_onnx_model(filepath_to)
        else:
            screen.warning("No ONNX files found in {}".format(ckpt_dir))
        
    @classmethod
    def train_main(cls):
        """Entrypoint for training.  
        Parses command-line arguments and starts training.
        """
        trainer = cls()
        trainer.launch()

        # Create model artifact for model.tar.gz
        parser = trainer.sagemaker_argparser()
        sage_args, unknown = parser.parse_known_args()
        if sage_args.save_model == 1:
            backend = os.getenv('COACH_BACKEND', 'tensorflow')
            if backend == 'tensorflow':
                trainer._save_tf_model()
            if backend == 'mxnet':
                trainer._save_onnx_model()


class SageMakerCoachLauncher(SageMakerCoachPresetLauncher):
    """
    Older version of the launcher that doesn't use preset, but instead effectively has a single preset built in.
    """

    def __init__(self):
        super().__init__()
        screen.warning("DEPRECATION WARNING: Please switch to SageMakerCoachPresetLauncher")
        #TODO: Remove this whole class when nobody's using it any more.

    def define_environment(self):
        return NotImplementedEror("Sub-class must define environment e.g. GymVectorEnvironment(level='your_module:YourClass')")

    def get_graph_manager_from_args(self, args):
        """Returns the GraphManager object for coach to use to train by calling improve()
        """
        # NOTE: TaskParameters are not configurable at this time.

        # Visualization
        vis_params = VisualizationParameters()
        self.config_visualization(vis_params)
        self.hyperparameters.apply_subset(vis_params, "vis_params.")

        # Schedule
        schedule_params = ScheduleParameters()
        self.config_schedule(schedule_params)
        self.hyperparameters.apply_subset(schedule_params, "schedule_params.")

        # Agent
        agent_params = self.define_agent()
        self.hyperparameters.apply_subset(agent_params, "agent_params.")

        # Environment
        env_params = self.define_environment()
        self.hyperparameters.apply_subset(env_params, "env_params.")

        graph_manager = BasicRLGraphManager(
            agent_params=agent_params,
            env_params=env_params,
            schedule_params=schedule_params,
            vis_params=vis_params,
        )

        return graph_manager

    def config_schedule(self, schedule_params):
        pass

    def define_agent(self):
        raise NotImplementedError("Subclass must create define_agent() method which returns an AgentParameters object. e.g.\n" \
            "   return rl_coach.agents.dqn_agent.DQNAgentParameters()");

    def config_visualization(self, vis_params):
        vis_params.dump_gifs = True
        vis_params.video_dump_methods = [SelectedPhaseOnlyDumpFilter(RunPhase.TEST), MaxDumpFilter()]
        vis_params.print_networks_summary = True
        return vis_params
