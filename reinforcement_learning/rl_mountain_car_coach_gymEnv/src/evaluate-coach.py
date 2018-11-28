from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher, CoachConfigurationList
import argparse
import os
import rl_coach
from rl_coach.base_parameters import Frameworks, TaskParameters
from rl_coach.core_types import EnvironmentSteps


def inplace_replace_in_file(filepath, old, new):
    with open(filepath, 'r') as f:
        contents = f.read()
    with open(filepath, 'w') as f:
        contents = contents.replace(old, new)
        f.write(contents)    
        

class MyLauncher(SageMakerCoachPresetLauncher):

    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return 'preset-cartpole-dqn'
    
    def start_single_threaded(self, task_parameters, graph_manager, args):
        """Override to use custom evaluate_steps, instead of infinite steps. Just evaluate.
        """
        graph_manager.agent_params.visualization.dump_csv = False # issues with CSV export in evaluation only
        graph_manager.create_graph(task_parameters)
        graph_manager.evaluate(EnvironmentSteps(args.evaluate_steps))
        graph_manager.close()
    
    def get_config_args(self, parser):
        """Overrides the default CLI parsing.
        Sets the configuration parameters for what a SageMaker run should do.
        Note, this does not support the "play" mode.
        """
        ### Parse Arguments
        # first, convert the parser to a Namespace object with all default values.
        empty_arg_list = []
        args, _ = parser.parse_known_args(args=empty_arg_list)
        parser = self.sagemaker_argparser()
        sage_args, unknown = parser.parse_known_args()

        ### Set Arguments
        args.preset = sage_args.RLCOACH_PRESET
        backend = os.getenv('COACH_BACKEND', 'tensorflow')
        args.framework = args.framework = Frameworks[backend]
        args.checkpoint_save_dir = None
        args.checkpoint_restore_dir = "/opt/ml/input/data/checkpoint"
        # Correct TensorFlow checkpoint file (https://github.com/tensorflow/tensorflow/issues/9146)
        if backend == "tensorflow":
            checkpoint_filepath = os.path.join(args.checkpoint_restore_dir, 'checkpoint')
            inplace_replace_in_file(checkpoint_filepath, "/opt/ml/output/data/checkpoint", ".")            
        # Override experiment_path used for outputs (note CSV not stored, see `start_single_threaded`).
        args.experiment_path = '/opt/ml/output/intermediate'
        rl_coach.logger.experiment_path = '/opt/ml/output/intermediate' # for gifs
        args.evaluate = True # not actually used, but must be set (see `evaluate_steps`)
        args.evaluate_steps = sage_args.evaluate_steps
        args.no_summary = True # so process doesn't hang at end
        # must be set
        self.hyperparameters = CoachConfigurationList()
        
        return args
    
    def sagemaker_argparser(self):
        """
        Expose only the CLI arguments that make sense in the SageMaker context.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--RLCOACH_PRESET',
                            help="(string) Name of the file with the RLCoach preset",
                            default=self.default_preset_name(),
                            type=str)
        parser.add_argument('--evaluate_steps',
                            help="(int) Number of evaluation steps to takr",
                            default=1000,
                            type=int)
        return parser    
    
    @classmethod
    def evaluate_main(cls):
        """Entrypoint for training.  
        Parses command-line arguments and starts training.
        """
        evaluator = cls()
        evaluator.launch()

    
if __name__ == '__main__':
    MyLauncher.evaluate_main()