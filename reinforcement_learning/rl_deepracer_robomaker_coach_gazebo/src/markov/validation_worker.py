import os
import sys

# Override os._exit with sys.exit for validation worker,
# so log_and_exit will call sys.exit actually when calling os._exit to exit the process.
# - There is hanging issue if the process exits with os._exit when exception is thrown from
#   tensorflow.
# - Also, we cannot replace os._exit with sys.exit in exception_handler.simapp_exit_gracefully
#   which is called by log_and_exit as os._exit is only way to fault the RoboMaker job, when
#   SimApp faults and exits.
#   - Otherwise, RoboMaker job ignores SimApp termination and runs until timeout instead of faulting
#     right away when SimApp faults and exits if SimApp exited other than os._exit.
os._exit = sys.exit
import argparse
import logging
import pickle

from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.training_agent_factory import create_training_agent
from markov.architecture.constants import Input
from markov.boto.s3.constants import MODEL_METADATA_S3_POSTFIX, ModelMetadataKeys
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.boto.s3.files.model_metadata import ModelMetadata
from markov.boto.s3.utils import get_s3_key
from markov.constants import BEST_CHECKPOINT, LAST_CHECKPOINT, SIMAPP_VERSION_2
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_VALIDATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps, RunPhase
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.logger import screen

logger = Logger(__name__, logging.INFO).get_logger()

SAMPLE_PICKLE_PATH = "/opt/ml/code/sample_data"
MODEL_METADATA_LOCAL_PATH = "model_metadata.json"
LOCAL_MODEL_DIR = "local_model_checkpoint"


def _validate(graph_manager, task_parameters, transitions, s3_bucket, s3_prefix, aws_region):
    checkpoint = graph_manager.data_store.params.checkpoint_dict["agent"]
    checkpoint_dir = task_parameters.checkpoint_restore_path
    graph_manager.data_store.wait_for_checkpoints()

    # validate last checkpoint
    last_model_checkpoint_name = (
        checkpoint.deepracer_checkpoint_json.get_deepracer_last_checkpoint()
    )
    if checkpoint.rl_coach_checkpoint.update(
        model_checkpoint_name=last_model_checkpoint_name,
        s3_kms_extra_args=utils.get_s3_kms_extra_args(),
    ):
        screen.log_title(" Validating Last Checkpoint: {}".format(last_model_checkpoint_name))
        # load the last rl coach checkpoint from store
        graph_manager.data_store.load_from_store()
        graph_manager.create_graph(task_parameters)
        graph_manager.phase = RunPhase.TEST
        screen.log_title(" Start emulate_act_on_trainer on Last Checkpoint")
        graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)
        screen.log_title(" emulate_act_on_trainer on Last Checkpoint completed!")
        # validate best checkpoint: Best checkpoint might not exist.
        best_model_checkpoint_name = (
            checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint()
        )
        if checkpoint.rl_coach_checkpoint.update(
            model_checkpoint_name=best_model_checkpoint_name,
            s3_kms_extra_args=utils.get_s3_kms_extra_args(),
        ):
            screen.log_title(" Validating Best Checkpoint: {}".format(best_model_checkpoint_name))
            # load the best rl coach checkpoint from store
            graph_manager.data_store.load_from_store()
            graph_manager.restore_checkpoint()
            screen.log_title(" Start emulate_act_on_trainer on Best Checkpoint")
            graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)
            screen.log_title(" emulate_act_on_trainer on Best Checkpoint completed!")
        else:
            screen.log_title(" No Best Checkpoint to validate.")

    else:
        screen.log_title(" Validating Last Checkpoint")
        # load the last rl coach checkpoint from store
        graph_manager.data_store.load_from_store()
        graph_manager.create_graph(task_parameters)
        graph_manager.phase = RunPhase.TEST
        screen.log_title(" Start emulate_act_on_trainer on Last Checkpoint ")
        graph_manager.emulate_act_on_trainer(EnvironmentSteps(1), transitions=transitions)
        screen.log_title(" Start emulate_act_on_trainer on Last Checkpoint completed!")
    screen.log_title(" Validation completed!")


def get_transition_data(observation_list):
    single_camera_sensor_list = [Input.OBSERVATION.value, Input.CAMERA.value]
    if any([sensor in single_camera_sensor_list for sensor in observation_list]):
        pickle_filename = Input.CAMERA.value.lower()
        if Input.LIDAR.value in observation_list or Input.SECTOR_LIDAR.value in observation_list:
            pickle_filename += "_" + Input.LIDAR.value.lower()
        pickle_filename += ".pkl"
    elif Input.STEREO.value in observation_list:
        pickle_filename = Input.STEREO.value.lower()
        if Input.LIDAR.value in observation_list or Input.SECTOR_LIDAR.value in observation_list:
            pickle_filename += "_" + Input.LIDAR.value.lower()
        pickle_filename += ".pkl"
    else:
        log_and_exit(
            "Sensor not supported: {}!".format(observation_list),
            SIMAPP_VALIDATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )

    pickle_path = os.path.join(SAMPLE_PICKLE_PATH, pickle_filename)
    with open(pickle_path, "rb") as in_f:
        return pickle.load(in_f)


# validate function below can be directly used by model validation container,
# if we fix preemptive termination with os._exit in log_and_exit
# or simapp_exit_gracefully when error/exception is raised.
def validate(s3_bucket, s3_prefix, aws_region):
    screen.set_use_colors(False)
    screen.log_title(" S3 bucket: {} \n S3 prefix: {}".format(s3_bucket, s3_prefix))

    # download model metadata
    model_metadata = ModelMetadata(
        bucket=s3_bucket,
        s3_key=get_s3_key(s3_prefix, MODEL_METADATA_S3_POSTFIX),
        region_name=aws_region,
        local_path=MODEL_METADATA_LOCAL_PATH,
    )

    # Create model local path
    os.makedirs(LOCAL_MODEL_DIR)

    try:
        # Handle backward compatibility
        model_metadata_info = model_metadata.get_model_metadata_info()
        observation_list = model_metadata_info[ModelMetadataKeys.SENSOR.value]
        version = model_metadata_info[ModelMetadataKeys.VERSION.value]
    except Exception as ex:
        log_and_exit(
            "Failed to parse model_metadata file: {}".format(ex),
            SIMAPP_VALIDATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )

    # Below get_transition_data function must called before create_training_agent function
    # to avoid 500 in case unsupported Sensor is received.
    # create_training_agent will exit with 500 if unsupported sensor is received,
    # and get_transition_data function below will exit with 400 if unsupported sensor is received.
    # We want to return 400 in model validation case if unsupported sensor is received.
    # Thus, call this get_transition_data function before create_traning_agent function!
    transitions = get_transition_data(observation_list)

    checkpoint = Checkpoint(
        bucket=s3_bucket,
        s3_prefix=s3_prefix,
        region_name=args.aws_region,
        agent_name="agent",
        checkpoint_dir=LOCAL_MODEL_DIR,
    )
    # make coach checkpoint compatible
    if version < SIMAPP_VERSION_2 and not checkpoint.rl_coach_checkpoint.is_compatible():
        checkpoint.rl_coach_checkpoint.make_compatible(checkpoint.syncfile_ready)
    # add checkpoint into checkpoint_dict
    checkpoint_dict = {"agent": checkpoint}

    agent_config = {
        "model_metadata": model_metadata,
        ConfigParams.CAR_CTRL_CONFIG.value: {
            ConfigParams.LINK_NAME_LIST.value: [],
            ConfigParams.VELOCITY_LIST.value: {},
            ConfigParams.STEERING_LIST.value: {},
            ConfigParams.CHANGE_START.value: None,
            ConfigParams.ALT_DIR.value: None,
            ConfigParams.MODEL_METADATA.value: model_metadata,
            ConfigParams.REWARD.value: None,
            ConfigParams.AGENT_NAME.value: "racecar",
        },
    }

    agent_list = list()
    agent_list.append(create_training_agent(agent_config))

    sm_hyperparams_dict = {}
    graph_manager, _ = get_graph_manager(
        hp_dict=sm_hyperparams_dict, agent_list=agent_list, run_phase_subject=None
    )

    ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager, ignore_lock=True)

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = LOCAL_MODEL_DIR
    _validate(
        graph_manager=graph_manager,
        task_parameters=task_parameters,
        transitions=transitions,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        aws_region=aws_region,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--s3_bucket", help="(string) S3 bucket", type=str)
    parser.add_argument("--s3_prefix", help="(string) S3 prefix", type=str)
    parser.add_argument("--aws_region", help="(string) AWS region", type=str)
    args = parser.parse_args()

    try:
        validate(s3_bucket=args.s3_bucket, s3_prefix=args.s3_prefix, aws_region=args.aws_region)
    except ValueError as err:
        if utils.is_user_error(err):
            log_and_exit(
                "User modified model/model_metadata: {}".format(err),
                SIMAPP_VALIDATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_400,
            )
        else:
            log_and_exit(
                "Validation worker value error: {}".format(err),
                SIMAPP_VALIDATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
    except Exception as ex:
        log_and_exit(
            "Validation worker exited with exception: {}".format(ex),
            SIMAPP_VALIDATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
