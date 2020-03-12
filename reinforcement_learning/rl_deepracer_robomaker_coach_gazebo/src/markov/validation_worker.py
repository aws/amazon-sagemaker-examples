import os
import argparse
import logging
import pickle
import shutil

from rl_coach.base_parameters import TaskParameters
from rl_coach.logger import screen
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.rollout_worker import wait_for_checkpoint
from rl_coach.core_types import EnvironmentSteps, RunPhase

from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.training_agent_factory import create_training_agent
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.utils_parse_model_metadata import parse_model_metadata
from markov.deepracer_exceptions import GenericValidatorException, GenericValidatorError
from markov.architecture.constants import Input

logger = utils.Logger(__name__, logging.INFO).get_logger()

SAMPLE_PICKLE_PATH = '/opt/ml/code/sample_data'


def _validate(graph_manager, task_parameters, transitions,
              s3_bucket, s3_prefix, aws_region):
    checkpoint_dir = task_parameters.checkpoint_restore_path
    wait_for_checkpoint(checkpoint_dir, graph_manager.data_store)

    if utils.do_model_selection(s3_bucket=s3_bucket,
                                s3_prefix=s3_prefix,
                                region=aws_region,
                                checkpoint_type=utils.LAST_CHECKPOINT):
        logger.info("Test Last Checkpoint: %s", utils.get_best_checkpoint(s3_bucket, s3_prefix, aws_region))
        graph_manager.create_graph(task_parameters)
        graph_manager.phase = RunPhase.TEST
        graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)
        logger.info("Test Best Checkpoint: %s", utils.get_last_checkpoint(s3_bucket, s3_prefix, aws_region))
        utils.do_model_selection(s3_bucket=s3_bucket,
                                 s3_prefix=s3_prefix,
                                 region=aws_region,
                                 checkpoint_type=utils.BEST_CHECKPOINT)
        graph_manager.data_store.load_from_store()
        graph_manager.restore_checkpoint()
        graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)
    else:
        logger.info("Test Last Checkpoint")
        graph_manager.create_graph(task_parameters)
        graph_manager.phase = RunPhase.TEST
        graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)


def get_transition_data(observation_list):
    single_camera_sensor_list = [Input.OBSERVATION.value, Input.CAMERA.value]
    if any([sensor in single_camera_sensor_list for sensor in observation_list]):
        pickle_filename = Input.CAMERA.value.lower()
        if Input.LIDAR.value in observation_list or Input.SECTOR_LIDAR.value in observation_list:
            pickle_filename += '_' + Input.LIDAR.value.lower()
        pickle_filename += '.pkl'
    elif Input.STEREO.value in observation_list:
        pickle_filename = Input.STEREO.value.lower()
        if Input.LIDAR.value in observation_list or Input.SECTOR_LIDAR.value in observation_list:
            pickle_filename += '_' + Input.LIDAR.value.lower()
        pickle_filename += '.pkl'
    else:
        GenericValidatorError("Sensor not supported: {}!".format(observation_list)).log_except_and_exit()

    pickle_path = os.path.join(SAMPLE_PICKLE_PATH, pickle_filename)
    with open(pickle_path, 'rb') as in_f:
        return pickle.load(in_f)


# validate function below can be directly used by model validation container,
# if we fix preemptive termination with os._exit in utils.log_and_exit
# or simapp_exit_gracefully when error/exception is raised.
def validate(s3_bucket, s3_prefix, custom_files_path, aws_region):
    screen.set_use_colors(False)
    logger.info("S3 bucket: %s \n S3 prefix: %s", s3_bucket, s3_prefix)

    if not os.path.exists(custom_files_path):
        os.makedirs(custom_files_path)
    else:
        GenericValidatorException("Custom Files Path already exists!").log_except_and_exit()

    s3_client = SageS3Client(bucket=s3_bucket,
                             s3_prefix=s3_prefix,
                             aws_region=aws_region)
    # Load the model metadata
    model_metadata_local_path = os.path.join(custom_files_path, 'model_metadata.json')
    utils.load_model_metadata(s3_client,
                              os.path.normpath("%s/model/model_metadata.json" % s3_prefix),
                              model_metadata_local_path)

    # Create model local path
    local_model_dir = os.path.join(custom_files_path, 'checkpoint')
    os.makedirs(local_model_dir)

    try:
        # Handle backward compatibility
        observation_list, _, version = parse_model_metadata(model_metadata_local_path)
    except Exception as ex:
        utils.log_and_exit("Failed to parse model_metadata file: {}".format(ex),
                           utils.SIMAPP_VALIDATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)

    transitions = get_transition_data(observation_list)

    if float(version) < float(utils.SIMAPP_VERSION) and \
            not utils.has_current_ckpnt_name(s3_bucket, s3_prefix, aws_region):
        utils.make_compatible(s3_bucket, s3_prefix, aws_region, SyncFiles.TRAINER_READY.value)

    agent_config = {'model_metadata': model_metadata_local_path,
                    ConfigParams.CAR_CTRL_CONFIG.value: {ConfigParams.LINK_NAME_LIST.value: [],
                                       ConfigParams.VELOCITY_LIST.value: {},
                                       ConfigParams.STEERING_LIST.value: {},
                                       ConfigParams.CHANGE_START.value: None,
                                       ConfigParams.ALT_DIR.value: None,
                                       ConfigParams.ACTION_SPACE_PATH.value: model_metadata_local_path,
                                       ConfigParams.REWARD.value: None,
                                       ConfigParams.AGENT_NAME.value: 'racecar'}}

    agent_list = list()
    agent_list.append(create_training_agent(agent_config))

    sm_hyperparams_dict = {}
    graph_manager, _ = get_graph_manager(hp_dict=sm_hyperparams_dict,
                                         agent_list=agent_list,
                                         run_phase_subject=None)

    ds_params_instance = S3BotoDataStoreParameters(aws_region=aws_region,
                                                   bucket_names={'agent': s3_bucket},
                                                   s3_folders={'agent': s3_prefix},
                                                   base_checkpoint_dir=local_model_dir
                                                   )

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager, ignore_lock=True)

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = local_model_dir
    _validate(graph_manager=graph_manager,
              task_parameters=task_parameters,
              transitions=transitions,
              s3_bucket=s3_bucket,
              s3_prefix=s3_prefix,
              aws_region=aws_region)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--s3_bucket',
                        help='(string) S3 bucket',
                        type=str)
    parser.add_argument('--s3_prefix',
                        help='(string) S3 prefix',
                        type=str)
    parser.add_argument('--aws_region',
                        help='(string) AWS region',
                        type=str)
    parser.add_argument('--custom_files_path',
                        help='(string) Custom Files Path',
                        type=str)

    args = parser.parse_args()

    try:
        validate(s3_bucket=args.s3_bucket,
                 s3_prefix=args.s3_prefix,
                 custom_files_path=args.custom_files_path,
                 aws_region=args.aws_region)
        shutil.rmtree(args.custom_files_path, ignore_errors=True)
    except ValueError as err:
        # folder deletion needs to happen every flows.
        # Since utils.log_and_exit uses os._exit, finally won't work.
        shutil.rmtree(args.custom_files_path, ignore_errors=True)
        if utils.is_error_bad_ckpnt(err):
            utils.log_and_exit("User modified model: {}".format(err),
                               utils.SIMAPP_VALIDATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_400)
        else:
            utils.log_and_exit("Validation worker value error: {}".format(err),
                               utils.SIMAPP_VALIDATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_500)
    except Exception as ex:
        # folder deletion needs to happen every flows.
        # Since utils.log_and_exit uses os._exit, finally won't work.
        shutil.rmtree(args.custom_files_path, ignore_errors=True)
        utils.log_and_exit("Validation worker exited with exception: {}".format(ex),
                           utils.SIMAPP_VALIDATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
