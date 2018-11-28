from rl_coach.agents.clipped_ppo_agent import ClippedPPOAgentParameters
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.base_parameters import VisualizationParameters, TaskParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps, RunPhase
from rl_coach import logger
import os
import argparse
import copy


def add_items_to_dict(target_dict, source_dict):
    updated_task_parameters = copy.copy(source_dict)
    updated_task_parameters.update(target_dict)
    return updated_task_parameters


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def evaluate(params):
    # file params
    experiment_path = os.path.join(params.output_data_dir)
    logger.experiment_path = os.path.join(experiment_path, 'evaluation')
    params.checkpoint_restore_dir = os.path.join(params.input_data_dir, 'checkpoint')
    checkpoint_file = os.path.join(params.checkpoint_restore_dir, 'checkpoint')

    inplace_change(checkpoint_file, "/opt/ml/output/data/checkpoint", ".")
    # Note that due to a tensorflow issue (https://github.com/tensorflow/tensorflow/issues/9146) we need to replace
    # the absolute path for the evaluation-from-a-checkpointed-model to work

    vis_params = VisualizationParameters()
    vis_params.dump_gifs = True

    task_params = TaskParameters(evaluate_only=True, experiment_path=logger.experiment_path)
    task_params.__dict__ = add_items_to_dict(task_params.__dict__, params.__dict__)

    graph_manager = BasicRLGraphManager(
        agent_params=ClippedPPOAgentParameters(),
        env_params=GymVectorEnvironment(level='TSP_env:TSPEasyEnv'),
        schedule_params=ScheduleParameters(),
        vis_params=vis_params
    )
    graph_manager = graph_manager.create_graph(task_parameters=task_params)
    graph_manager.evaluate(EnvironmentSteps(5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # consumes the hyper-parameters
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--input_data_dir', type=str, default='/opt/ml/input/data/')
    parser.add_argument('--output_data_dir', type=str, default='/opt/ml/output/data/')
    params, unknown = parser.parse_known_args()
    evaluate(params)
