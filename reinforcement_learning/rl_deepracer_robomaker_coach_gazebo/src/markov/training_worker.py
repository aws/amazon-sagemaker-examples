import argparse
import json
import logging
import math
import os
import shutil

import botocore
import numpy as np
import tensorflow as tf
from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.training_agent_factory import create_training_agent
from markov.boto.s3.constants import (
    FROZEN_HEAD_OUTPUT_GRAPH_FORMAT_MAPPING,
    HYPERPARAMETER_S3_POSTFIX,
    MODEL_METADATA_LOCAL_PATH_FORMAT,
    MODEL_METADATA_S3_POSTFIX,
    ModelMetadataKeys,
    TrainingAlgorithm,
)
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.boto.s3.files.hyperparameters import Hyperparameters
from markov.boto.s3.files.ip_config import IpConfig
from markov.boto.s3.files.model_metadata import ModelMetadata
from markov.boto.s3.s3_client import S3Client
from markov.boto.s3.utils import get_s3_key
from markov.constants import SIMAPP_VERSION_2, TRAINING_WORKER_PROFILER_PATH
from markov.deepracer_memory import DeepRacerRedisPubSubMemoryBackendParameters
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_TRAINING_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from markov.samples.sample_collector import SampleCollector
from rl_coach import core_types
from rl_coach.base_parameters import DistributedCoachSynchronizationType, RunType, TaskParameters
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.logger import screen
from rl_coach.utils import short_dynamic_import

tf.logging.set_verbosity(tf.logging.DEBUG)

logger = Logger(__name__, logging.INFO).get_logger()

PRETRAINED_MODEL_DIR = "./pretrained_checkpoint"
SM_MODEL_OUTPUT_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

IS_PROFILER_ON, PROFILER_S3_BUCKET, PROFILER_S3_PREFIX = utils.get_sagemaker_profiler_env()


def training_worker(
    graph_manager, task_parameters, user_batch_size, user_episode_per_rollout, training_algorithm
):
    try:
        # initialize graph
        graph_manager.create_graph(task_parameters)

        # save initial checkpoint
        graph_manager.save_checkpoint()

        # training loop
        steps = 0

        graph_manager.setup_memory_backend()
        graph_manager.signal_ready()

        # To handle SIGTERM
        door_man = utils.DoorMan()

        while steps < graph_manager.improve_steps.num_steps:
            # Collect profiler information only IS_PROFILER_ON is true
            with utils.Profiler(
                s3_bucket=PROFILER_S3_BUCKET,
                s3_prefix=PROFILER_S3_PREFIX,
                output_local_path=TRAINING_WORKER_PROFILER_PATH,
                enable_profiling=IS_PROFILER_ON,
            ):
                graph_manager.phase = core_types.RunPhase.TRAIN
                graph_manager.fetch_from_worker(
                    graph_manager.agent_params.algorithm.num_consecutive_playing_steps
                )
                graph_manager.phase = core_types.RunPhase.UNDEFINED

                episodes_in_rollout = graph_manager.memory_backend.get_total_episodes_in_rollout()

                for level in graph_manager.level_managers:
                    for agent in level.agents.values():
                        agent.ap.algorithm.num_consecutive_playing_steps.num_steps = (
                            episodes_in_rollout
                        )
                        agent.ap.algorithm.num_steps_between_copying_online_weights_to_target.num_steps = (
                            episodes_in_rollout
                        )

                # TODO: Refactor the flow to remove conditional checks for specific algorithms
                # ------------------------sac only---------------------------------------------
                if training_algorithm == TrainingAlgorithm.SAC.value:
                    rollout_steps = graph_manager.memory_backend.get_rollout_steps()

                    # NOTE: you can train even more iterations than rollout_steps by increasing the number below for SAC
                    agent.ap.algorithm.num_consecutive_training_steps = list(
                        rollout_steps.values()
                    )[
                        0
                    ]  # rollout_steps[agent]
                # -------------------------------------------------------------------------------
                if graph_manager.should_train():
                    # Make sure we have enough data for the requested batches
                    rollout_steps = graph_manager.memory_backend.get_rollout_steps()
                    if any(rollout_steps.values()) <= 0:
                        log_and_exit(
                            "No rollout data retrieved from the rollout worker",
                            SIMAPP_TRAINING_WORKER_EXCEPTION,
                            SIMAPP_EVENT_ERROR_CODE_500,
                        )

                    # TODO: Refactor the flow to remove conditional checks for specific algorithms
                    # DH: for SAC, check if experience replay memory has enough transitions
                    if training_algorithm == TrainingAlgorithm.SAC.value:
                        replay_mem_size = min(
                            [
                                agent.memory.num_transitions()
                                for level in graph_manager.level_managers
                                for agent in level.agents.values()
                            ]
                        )
                        episode_batch_size = (
                            user_batch_size
                            if replay_mem_size > user_batch_size
                            else 2 ** math.floor(math.log(min(rollout_steps.values()), 2))
                        )
                    else:
                        episode_batch_size = (
                            user_batch_size
                            if min(rollout_steps.values()) > user_batch_size
                            else 2 ** math.floor(math.log(min(rollout_steps.values()), 2))
                        )
                    # Set the batch size to the closest power of 2 such that we have at least two batches, this prevents coach from crashing
                    # as  batch size less than 2 causes the batch list to become a scalar which causes an exception
                    for level in graph_manager.level_managers:
                        for agent in level.agents.values():
                            for net_key in agent.ap.network_wrappers:
                                agent.ap.network_wrappers[net_key].batch_size = episode_batch_size

                    steps += 1

                    graph_manager.phase = core_types.RunPhase.TRAIN
                    graph_manager.train()
                    graph_manager.phase = core_types.RunPhase.UNDEFINED

                    # Check for Nan's in all agents
                    rollout_has_nan = False
                    for level in graph_manager.level_managers:
                        for agent in level.agents.values():
                            if np.isnan(agent.loss.get_mean()):
                                rollout_has_nan = True
                    if rollout_has_nan:
                        log_and_exit(
                            "NaN detected in loss function, aborting training.",
                            SIMAPP_TRAINING_WORKER_EXCEPTION,
                            SIMAPP_EVENT_ERROR_CODE_500,
                        )

                    if (
                        graph_manager.agent_params.algorithm.distributed_coach_synchronization_type
                        == DistributedCoachSynchronizationType.SYNC
                    ):
                        graph_manager.save_checkpoint()
                    else:
                        graph_manager.occasionally_save_checkpoint()

                    # Clear any data stored in signals that is no longer necessary
                    graph_manager.reset_internal_state()

                for level in graph_manager.level_managers:
                    for agent in level.agents.values():
                        agent.ap.algorithm.num_consecutive_playing_steps.num_steps = (
                            user_episode_per_rollout
                        )
                        agent.ap.algorithm.num_steps_between_copying_online_weights_to_target.num_steps = (
                            user_episode_per_rollout
                        )

                if door_man.terminate_now:
                    log_and_exit(
                        "Received SIGTERM. Checkpointing before exiting.",
                        SIMAPP_TRAINING_WORKER_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_500,
                    )
                    graph_manager.save_checkpoint()
                    break

    except ValueError as err:
        if utils.is_user_error(err):
            log_and_exit(
                "User modified model: {}".format(err),
                SIMAPP_TRAINING_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        else:
            log_and_exit(
                "An error occured while training: {}".format(err),
                SIMAPP_TRAINING_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
    except Exception as ex:
        log_and_exit(
            "An error occured while training: {}".format(ex),
            SIMAPP_TRAINING_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    finally:
        graph_manager.data_store.upload_finished_file()


def main():
    screen.set_use_colors(False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pk",
        "--preset_s3_key",
        help="(string) Name of a preset to download from S3",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-ek",
        "--environment_s3_key",
        help="(string) Name of an environment file to download from S3",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model_metadata_s3_key",
        help="(string) Model Metadata File S3 Key",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        help="(string) Path to a folder containing a checkpoint to write the model to.",
        type=str,
        default="./checkpoint",
    )
    parser.add_argument(
        "--pretrained_checkpoint_dir",
        help="(string) Path to a folder for downloading a pre-trained model",
        type=str,
        default=PRETRAINED_MODEL_DIR,
    )
    parser.add_argument(
        "--s3_bucket",
        help="(string) S3 bucket",
        type=str,
        default=os.environ.get("SAGEMAKER_SHARED_S3_BUCKET_PATH", "gsaur-test"),
    )
    parser.add_argument("--s3_prefix", help="(string) S3 prefix", type=str, default="sagemaker")
    parser.add_argument(
        "--framework", help="(string) tensorflow or mxnet", type=str, default="tensorflow"
    )
    parser.add_argument(
        "--pretrained_s3_bucket", help="(string) S3 bucket for pre-trained model", type=str
    )
    parser.add_argument(
        "--pretrained_s3_prefix",
        help="(string) S3 prefix for pre-trained model",
        type=str,
        default="sagemaker",
    )
    parser.add_argument(
        "--aws_region",
        help="(string) AWS region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
    )

    args, _ = parser.parse_known_args()

    s3_client = S3Client(region_name=args.aws_region, max_retry_attempts=0)

    # download model metadata
    # TODO: replace 'agent' with name of each agent
    model_metadata_download = ModelMetadata(
        bucket=args.s3_bucket,
        s3_key=args.model_metadata_s3_key,
        region_name=args.aws_region,
        local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format("agent"),
    )
    model_metadata_info = model_metadata_download.get_model_metadata_info()
    network_type = model_metadata_info[ModelMetadataKeys.NEURAL_NETWORK.value]
    version = model_metadata_info[ModelMetadataKeys.VERSION.value]

    # upload model metadata
    model_metadata_upload = ModelMetadata(
        bucket=args.s3_bucket,
        s3_key=get_s3_key(args.s3_prefix, MODEL_METADATA_S3_POSTFIX),
        region_name=args.aws_region,
        local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format("agent"),
    )
    model_metadata_upload.persist(s3_kms_extra_args=utils.get_s3_kms_extra_args())

    shutil.copy2(model_metadata_download.local_path, SM_MODEL_OUTPUT_DIR)

    success_custom_preset = False
    if args.preset_s3_key:
        preset_local_path = "./markov/presets/preset.py"
        try:
            s3_client.download_file(
                bucket=args.s3_bucket, s3_key=args.preset_s3_key, local_path=preset_local_path
            )
            success_custom_preset = True
        except botocore.exceptions.ClientError:
            pass
        if not success_custom_preset:
            logger.info("Could not download the preset file. Using the default DeepRacer preset.")
        else:
            preset_location = "markov.presets.preset:graph_manager"
            graph_manager = short_dynamic_import(preset_location, ignore_module_case=True)
            s3_client.upload_file(
                bucket=args.s3_bucket,
                s3_key=os.path.normpath("%s/presets/preset.py" % args.s3_prefix),
                local_path=preset_local_path,
                s3_kms_extra_args=utils.get_s3_kms_extra_args(),
            )
            if success_custom_preset:
                logger.info("Using preset: %s" % args.preset_s3_key)

    if not success_custom_preset:
        params_blob = os.environ.get("SM_TRAINING_ENV", "")
        if params_blob:
            params = json.loads(params_blob)
            sm_hyperparams_dict = params["hyperparameters"]
        else:
            sm_hyperparams_dict = {}

        #! TODO each agent should have own config
        agent_config = {
            "model_metadata": model_metadata_download,
            ConfigParams.CAR_CTRL_CONFIG.value: {
                ConfigParams.LINK_NAME_LIST.value: [],
                ConfigParams.VELOCITY_LIST.value: {},
                ConfigParams.STEERING_LIST.value: {},
                ConfigParams.CHANGE_START.value: None,
                ConfigParams.ALT_DIR.value: None,
                ConfigParams.MODEL_METADATA.value: model_metadata_download,
                ConfigParams.REWARD.value: None,
                ConfigParams.AGENT_NAME.value: "racecar",
            },
        }

        agent_list = list()
        agent_list.append(create_training_agent(agent_config))

        graph_manager, robomaker_hyperparams_json = get_graph_manager(
            hp_dict=sm_hyperparams_dict,
            agent_list=agent_list,
            run_phase_subject=None,
            run_type=str(RunType.TRAINER),
        )

        # Upload hyperparameters to SageMaker shared s3 bucket
        hyperparameters = Hyperparameters(
            bucket=args.s3_bucket,
            s3_key=get_s3_key(args.s3_prefix, HYPERPARAMETER_S3_POSTFIX),
            region_name=args.aws_region,
        )
        hyperparameters.persist(
            hyperparams_json=robomaker_hyperparams_json,
            s3_kms_extra_args=utils.get_s3_kms_extra_args(),
        )

        # Attach sample collector to graph_manager only if sample count > 0
        max_sample_count = int(sm_hyperparams_dict.get("max_sample_count", 0))
        if max_sample_count > 0:
            sample_collector = SampleCollector(
                bucket=args.s3_bucket,
                s3_prefix=args.s3_prefix,
                region_name=args.aws_region,
                max_sample_count=max_sample_count,
                sampling_frequency=int(sm_hyperparams_dict.get("sampling_frequency", 1)),
            )
            graph_manager.sample_collector = sample_collector

    # persist IP config from sagemaker to s3
    ip_config = IpConfig(
        bucket=args.s3_bucket, s3_prefix=args.s3_prefix, region_name=args.aws_region
    )
    ip_config.persist(s3_kms_extra_args=utils.get_s3_kms_extra_args())

    training_algorithm = model_metadata_download.training_algorithm
    output_head_format = FROZEN_HEAD_OUTPUT_GRAPH_FORMAT_MAPPING[training_algorithm]

    use_pretrained_model = args.pretrained_s3_bucket and args.pretrained_s3_prefix
    # Handle backward compatibility
    if use_pretrained_model:
        # checkpoint s3 instance for pretrained model
        # TODO: replace 'agent' for multiagent training
        checkpoint = Checkpoint(
            bucket=args.pretrained_s3_bucket,
            s3_prefix=args.pretrained_s3_prefix,
            region_name=args.aws_region,
            agent_name="agent",
            checkpoint_dir=args.pretrained_checkpoint_dir,
            output_head_format=output_head_format,
        )
        # make coach checkpoint compatible
        if version < SIMAPP_VERSION_2 and not checkpoint.rl_coach_checkpoint.is_compatible():
            checkpoint.rl_coach_checkpoint.make_compatible(checkpoint.syncfile_ready)
        # get best model checkpoint string
        model_checkpoint_name = checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint()
        # Select the best checkpoint model by uploading rl coach .coach_checkpoint file
        checkpoint.rl_coach_checkpoint.update(
            model_checkpoint_name=model_checkpoint_name,
            s3_kms_extra_args=utils.get_s3_kms_extra_args(),
        )
        # add checkpoint into checkpoint_dict
        checkpoint_dict = {"agent": checkpoint}
        # load pretrained model
        ds_params_instance_pretrained = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)
        data_store_pretrained = S3BotoDataStore(ds_params_instance_pretrained, graph_manager, True)
        data_store_pretrained.load_from_store()

    memory_backend_params = DeepRacerRedisPubSubMemoryBackendParameters(
        redis_address="localhost",
        redis_port=6379,
        run_type=str(RunType.TRAINER),
        channel=args.s3_prefix,
        network_type=network_type,
    )

    graph_manager.memory_backend_params = memory_backend_params

    # checkpoint s3 instance for training model
    checkpoint = Checkpoint(
        bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region_name=args.aws_region,
        agent_name="agent",
        checkpoint_dir=args.checkpoint_dir,
        output_head_format=output_head_format,
    )
    checkpoint_dict = {"agent": checkpoint}
    ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

    graph_manager.data_store_params = ds_params_instance

    graph_manager.data_store = S3BotoDataStore(ds_params_instance, graph_manager)

    task_parameters = TaskParameters()
    task_parameters.experiment_path = SM_MODEL_OUTPUT_DIR
    task_parameters.checkpoint_save_secs = 20
    if use_pretrained_model:
        task_parameters.checkpoint_restore_path = args.pretrained_checkpoint_dir
    task_parameters.checkpoint_save_dir = args.checkpoint_dir

    training_worker(
        graph_manager=graph_manager,
        task_parameters=task_parameters,
        user_batch_size=json.loads(robomaker_hyperparams_json)["batch_size"],
        user_episode_per_rollout=json.loads(robomaker_hyperparams_json)[
            "num_episodes_between_training"
        ],
        training_algorithm=training_algorithm,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log_and_exit(
            "Training worker exited with exception: {}".format(ex),
            SIMAPP_TRAINING_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
