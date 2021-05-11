"""This module is responsible for launching evaluation jobs"""
import argparse
import json
import logging
import os
import time
from threading import Thread

import rospy
from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import (
    create_bot_cars_agent,
    create_obstacles_agent,
    create_rollout_agent,
)
from markov.agents.utils import RunPhaseSubject
from markov.boto.s3.constants import (
    CAMERA_45DEGREE_LOCAL_PATH_FORMAT,
    CAMERA_PIP_MP4_LOCAL_PATH_FORMAT,
    CAMERA_TOPVIEW_LOCAL_PATH_FORMAT,
    MODEL_METADATA_LOCAL_PATH_FORMAT,
    MODEL_METADATA_S3_POSTFIX,
    SIMTRACE_EVAL_LOCAL_PATH_FORMAT,
    ModelMetadataKeys,
    SimtraceVideoNames,
)
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.boto.s3.files.model_metadata import ModelMetadata
from markov.boto.s3.files.simtrace_video import SimtraceVideo
from markov.boto.s3.utils import get_s3_key
from markov.camera_utils import configure_camera
from markov.constants import DEFAULT_PARK_POSITION, ROLLOUT_WORKER_PROFILER_PATH, SIMAPP_VERSION_2
from markov.defaults import reward_function
from markov.environments.constants import LINK_NAMES, STEERING_TOPICS, VELOCITY_TOPICS
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericRolloutError, GenericRolloutException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.metrics.constants import MetricsS3Keys
from markov.metrics.iteration_data import IterationData
from markov.metrics.s3_metrics import EvalMetrics
from markov.reset.constants import RaceType
from markov.rollout_utils import (
    PhaseObserver,
    configure_environment_randomizer,
    get_robomaker_profiler_env,
    signal_robomaker_markov_package_ready,
)
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from markov.track_geom.track_data import TrackData
from markov.track_geom.utils import get_start_positions
from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.data_stores.data_store import SyncFiles
from std_srvs.srv import Empty, EmptyRequest

logger = Logger(__name__, logging.INFO).get_logger()

MIN_RESET_COUNT = 10000  # TODO: change when console passes float("inf")

IS_PROFILER_ON, PROFILER_S3_BUCKET, PROFILER_S3_PREFIX = get_robomaker_profiler_env()


def evaluation_worker(
    graph_manager,
    number_of_trials,
    task_parameters,
    simtrace_video_s3_writers,
    is_continuous,
    park_positions,
    race_type,
    pause_physics,
    unpause_physics,
):
    """Evaluation worker function

    Arguments:
        graph_manager(MultiAgentGraphManager): Graph manager of multiagent graph manager
        number_of_trials(int): Number of trails you want to run the evaluation
        task_parameters(TaskParameters): Information of the checkpoint, gpu/cpu,
            framework etc of rlcoach
        simtrace_video_s3_writers(list): Information to upload to the S3 bucket all the simtrace and mp4
        is_continuous(bool): The termination condition for the car
        park_positions(list of tuple): list of (x, y) for cars to park at
        race_type (str): race type
    """
    # Collect profiler information only IS_PROFILER_ON is true
    with utils.Profiler(
        s3_bucket=PROFILER_S3_BUCKET,
        s3_prefix=PROFILER_S3_PREFIX,
        output_local_path=ROLLOUT_WORKER_PROFILER_PATH,
        enable_profiling=IS_PROFILER_ON,
    ):
        subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic = list(), list()
        subscribe_to_save_mp4, unsubscribe_from_save_mp4 = list(), list()
        for agent_param in graph_manager.agents_params:
            racecar_name = (
                "racecar"
                if len(agent_param.name.split("_")) == 1
                else "racecar_{}".format(agent_param.name.split("_")[1])
            )
            subscribe_to_save_mp4_topic.append(
                "/{}/save_mp4/subscribe_to_save_mp4".format(racecar_name)
            )
            unsubscribe_from_save_mp4_topic.append(
                "/{}/save_mp4/unsubscribe_from_save_mp4".format(racecar_name)
            )
        graph_manager.data_store.wait_for_checkpoints()
        graph_manager.data_store.modify_checkpoint_variables()

        # wait for the required cancel services to become available
        if race_type != RaceType.F1.value:
            # TODO: Since we are not running Grand Prix in RoboMaker,
            # we are opting out from waiting for RoboMaker's cancel job service
            # in case of Grand Prix execution.
            # Otherwise, SimApp will hang as service will never come alive.
            #
            # If we don't depend on RoboMaker anymore in the future,
            # we need to remove below line, or do a better job to figure out
            # whether we are running on RoboMaker or not to decide whether
            # we should wait for below service or not.
            rospy.wait_for_service("/robomaker/job/cancel")

        # Make the clients that will allow us to pause and unpause the physics
        rospy.wait_for_service("/gazebo/pause_physics_dr")
        rospy.wait_for_service("/gazebo/unpause_physics_dr")
        pause_physics = ServiceProxyWrapper("/gazebo/pause_physics_dr", Empty)
        unpause_physics = ServiceProxyWrapper("/gazebo/unpause_physics_dr", Empty)

        for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
            rospy.wait_for_service(mp4_sub)
            rospy.wait_for_service(mp4_unsub)
        for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
            subscribe_to_save_mp4.append(ServiceProxyWrapper(mp4_sub, Empty))
            unsubscribe_from_save_mp4.append(
                Thread(target=ServiceProxyWrapper(mp4_unsub, Empty), args=(EmptyRequest(),))
            )

        graph_manager.create_graph(
            task_parameters=task_parameters,
            stop_physics=pause_physics,
            start_physics=unpause_physics,
            empty_service_call=EmptyRequest,
        )
        logger.info("Graph manager successfully created the graph: Unpausing physics")
        unpause_physics(EmptyRequest())

        is_save_mp4_enabled = rospy.get_param("MP4_S3_BUCKET", None)
        if is_save_mp4_enabled:
            for subscribe_mp4 in subscribe_to_save_mp4:
                subscribe_mp4(EmptyRequest())

        configure_environment_randomizer()
        track_data = TrackData.get_instance()

        # Before each evaluation episode (single lap for non-continuous race and complete race for
        # continuous race), a new copy of park_positions needs to be loaded into track_data because
        # a park position will be pop from park_positions when a racer car need to be parked.
        if is_continuous:
            track_data.park_positions = park_positions
            graph_manager.evaluate(EnvironmentSteps(1))
        else:
            for _ in range(number_of_trials):
                track_data.park_positions = park_positions
                graph_manager.evaluate(EnvironmentSteps(1))
        if is_save_mp4_enabled:
            for unsubscribe_mp4 in unsubscribe_from_save_mp4:
                unsubscribe_mp4.start()
            for unsubscribe_mp4 in unsubscribe_from_save_mp4:
                unsubscribe_mp4.join()
        # upload simtrace and mp4 into s3 bucket
        for s3_writer in simtrace_video_s3_writers:
            s3_writer.persist(utils.get_s3_kms_extra_args())
        time.sleep(1)
        pause_physics(EmptyRequest())

    if race_type != RaceType.F1.value:
        # Close the down the job
        utils.cancel_simulation_job()


def main():
    """Main function for evaluation worker"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--preset",
        help="(string) Name of a preset to run \
                             (class name from the 'presets' directory.)",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--s3_bucket",
        help="list(string) S3 bucket",
        type=str,
        nargs="+",
        default=rospy.get_param("MODEL_S3_BUCKET", ["gsaur-test"]),
    )
    parser.add_argument(
        "--s3_prefix",
        help="list(string) S3 prefix",
        type=str,
        nargs="+",
        default=rospy.get_param("MODEL_S3_PREFIX", ["sagemaker"]),
    )
    parser.add_argument(
        "--aws_region",
        help="(string) AWS region",
        type=str,
        default=rospy.get_param("AWS_REGION", "us-east-1"),
    )
    parser.add_argument(
        "--number_of_trials",
        help="(integer) Number of trials",
        type=int,
        default=int(rospy.get_param("NUMBER_OF_TRIALS", 10)),
    )
    parser.add_argument(
        "-c",
        "--local_model_directory",
        help="(string) Path to a folder containing a checkpoint \
                             to restore the model from.",
        type=str,
        default="./checkpoint",
    )
    parser.add_argument(
        "--number_of_resets",
        help="(integer) Number of resets",
        type=int,
        default=int(rospy.get_param("NUMBER_OF_RESETS", 0)),
    )
    parser.add_argument(
        "--penalty_seconds",
        help="(float) penalty second",
        type=float,
        default=float(rospy.get_param("PENALTY_SECONDS", 2.0)),
    )
    parser.add_argument(
        "--job_type",
        help="(string) job type",
        type=str,
        default=rospy.get_param("JOB_TYPE", "EVALUATION"),
    )
    parser.add_argument(
        "--is_continuous",
        help="(boolean) is continous after lap completion",
        type=bool,
        default=utils.str2bool(rospy.get_param("IS_CONTINUOUS", False)),
    )
    parser.add_argument(
        "--race_type",
        help="(string) Race type",
        type=str,
        default=rospy.get_param("RACE_TYPE", "TIME_TRIAL"),
    )
    parser.add_argument(
        "--off_track_penalty",
        help="(float) off track penalty second",
        type=float,
        default=float(rospy.get_param("OFF_TRACK_PENALTY", 2.0)),
    )
    parser.add_argument(
        "--collision_penalty",
        help="(float) collision penalty second",
        type=float,
        default=float(rospy.get_param("COLLISION_PENALTY", 5.0)),
    )

    args = parser.parse_args()
    arg_s3_bucket = args.s3_bucket
    arg_s3_prefix = args.s3_prefix
    logger.info("S3 bucket: %s \n S3 prefix: %s", arg_s3_bucket, arg_s3_prefix)

    metrics_s3_buckets = rospy.get_param("METRICS_S3_BUCKET")
    metrics_s3_object_keys = rospy.get_param("METRICS_S3_OBJECT_KEY")

    arg_s3_bucket, arg_s3_prefix = utils.force_list(arg_s3_bucket), utils.force_list(arg_s3_prefix)
    metrics_s3_buckets = utils.force_list(metrics_s3_buckets)
    metrics_s3_object_keys = utils.force_list(metrics_s3_object_keys)

    validate_list = [arg_s3_bucket, arg_s3_prefix, metrics_s3_buckets, metrics_s3_object_keys]

    simtrace_s3_bucket = rospy.get_param("SIMTRACE_S3_BUCKET", None)
    mp4_s3_bucket = rospy.get_param("MP4_S3_BUCKET", None)
    if simtrace_s3_bucket:
        simtrace_s3_object_prefix = rospy.get_param("SIMTRACE_S3_PREFIX")
        simtrace_s3_bucket = utils.force_list(simtrace_s3_bucket)
        simtrace_s3_object_prefix = utils.force_list(simtrace_s3_object_prefix)
        validate_list.extend([simtrace_s3_bucket, simtrace_s3_object_prefix])
    if mp4_s3_bucket:
        mp4_s3_object_prefix = rospy.get_param("MP4_S3_OBJECT_PREFIX")
        mp4_s3_bucket = utils.force_list(mp4_s3_bucket)
        mp4_s3_object_prefix = utils.force_list(mp4_s3_object_prefix)
        validate_list.extend([mp4_s3_bucket, mp4_s3_object_prefix])

    if not all([lambda x: len(x) == len(validate_list[0]), validate_list]):
        log_and_exit(
            "Eval worker error: Incorrect arguments passed: {}".format(validate_list),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
    if args.number_of_resets != 0 and args.number_of_resets < MIN_RESET_COUNT:
        raise GenericRolloutException("number of resets is less than {}".format(MIN_RESET_COUNT))

    # Instantiate Cameras
    if len(arg_s3_bucket) == 1:
        configure_camera(namespaces=["racecar"])
    else:
        configure_camera(
            namespaces=[
                "racecar_{}".format(str(agent_index)) for agent_index in range(len(arg_s3_bucket))
            ]
        )

    agent_list = list()
    s3_bucket_dict = dict()
    s3_prefix_dict = dict()
    checkpoint_dict = dict()
    simtrace_video_s3_writers = []
    start_positions = get_start_positions(len(arg_s3_bucket))
    done_condition = utils.str_to_done_condition(rospy.get_param("DONE_CONDITION", any))
    park_positions = utils.pos_2d_str_to_list(rospy.get_param("PARK_POSITIONS", []))
    # if not pass in park positions for all done condition case, use default
    if not park_positions:
        park_positions = [DEFAULT_PARK_POSITION for _ in arg_s3_bucket]
    for agent_index, _ in enumerate(arg_s3_bucket):
        agent_name = "agent" if len(arg_s3_bucket) == 1 else "agent_{}".format(str(agent_index))
        racecar_name = (
            "racecar" if len(arg_s3_bucket) == 1 else "racecar_{}".format(str(agent_index))
        )
        s3_bucket_dict[agent_name] = arg_s3_bucket[agent_index]
        s3_prefix_dict[agent_name] = arg_s3_prefix[agent_index]

        # download model metadata
        model_metadata = ModelMetadata(
            bucket=arg_s3_bucket[agent_index],
            s3_key=get_s3_key(arg_s3_prefix[agent_index], MODEL_METADATA_S3_POSTFIX),
            region_name=args.aws_region,
            local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format(agent_name),
        )
        model_metadata_info = model_metadata.get_model_metadata_info()
        version = model_metadata_info[ModelMetadataKeys.VERSION.value]

        # checkpoint s3 instance
        checkpoint = Checkpoint(
            bucket=arg_s3_bucket[agent_index],
            s3_prefix=arg_s3_prefix[agent_index],
            region_name=args.aws_region,
            agent_name=agent_name,
            checkpoint_dir=args.local_model_directory,
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

        checkpoint_dict[agent_name] = checkpoint

        agent_config = {
            "model_metadata": model_metadata,
            ConfigParams.CAR_CTRL_CONFIG.value: {
                ConfigParams.LINK_NAME_LIST.value: [
                    link_name.replace("racecar", racecar_name) for link_name in LINK_NAMES
                ],
                ConfigParams.VELOCITY_LIST.value: [
                    velocity_topic.replace("racecar", racecar_name)
                    for velocity_topic in VELOCITY_TOPICS
                ],
                ConfigParams.STEERING_LIST.value: [
                    steering_topic.replace("racecar", racecar_name)
                    for steering_topic in STEERING_TOPICS
                ],
                ConfigParams.CHANGE_START.value: utils.str2bool(
                    rospy.get_param("CHANGE_START_POSITION", False)
                ),
                ConfigParams.ALT_DIR.value: utils.str2bool(
                    rospy.get_param("ALTERNATE_DRIVING_DIRECTION", False)
                ),
                ConfigParams.MODEL_METADATA.value: model_metadata,
                ConfigParams.REWARD.value: reward_function,
                ConfigParams.AGENT_NAME.value: racecar_name,
                ConfigParams.VERSION.value: version,
                ConfigParams.NUMBER_OF_RESETS.value: args.number_of_resets,
                ConfigParams.PENALTY_SECONDS.value: args.penalty_seconds,
                ConfigParams.NUMBER_OF_TRIALS.value: args.number_of_trials,
                ConfigParams.IS_CONTINUOUS.value: args.is_continuous,
                ConfigParams.RACE_TYPE.value: args.race_type,
                ConfigParams.COLLISION_PENALTY.value: args.collision_penalty,
                ConfigParams.OFF_TRACK_PENALTY.value: args.off_track_penalty,
                ConfigParams.START_POSITION.value: start_positions[agent_index],
                ConfigParams.DONE_CONDITION.value: done_condition,
            },
        }

        metrics_s3_config = {
            MetricsS3Keys.METRICS_BUCKET.value: metrics_s3_buckets[agent_index],
            MetricsS3Keys.METRICS_KEY.value: metrics_s3_object_keys[agent_index],
            # Replaced rospy.get_param('AWS_REGION') to be equal to the argument being passed
            # or default argument set
            MetricsS3Keys.REGION.value: args.aws_region,
        }
        aws_region = rospy.get_param("AWS_REGION", args.aws_region)

        if simtrace_s3_bucket:
            simtrace_video_s3_writers.append(
                SimtraceVideo(
                    upload_type=SimtraceVideoNames.SIMTRACE_EVAL.value,
                    bucket=simtrace_s3_bucket[agent_index],
                    s3_prefix=simtrace_s3_object_prefix[agent_index],
                    region_name=aws_region,
                    local_path=SIMTRACE_EVAL_LOCAL_PATH_FORMAT.format(agent_name),
                )
            )
        if mp4_s3_bucket:
            simtrace_video_s3_writers.extend(
                [
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.PIP.value,
                        bucket=mp4_s3_bucket[agent_index],
                        s3_prefix=mp4_s3_object_prefix[agent_index],
                        region_name=aws_region,
                        local_path=CAMERA_PIP_MP4_LOCAL_PATH_FORMAT.format(agent_name),
                    ),
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.DEGREE45.value,
                        bucket=mp4_s3_bucket[agent_index],
                        s3_prefix=mp4_s3_object_prefix[agent_index],
                        region_name=aws_region,
                        local_path=CAMERA_45DEGREE_LOCAL_PATH_FORMAT.format(agent_name),
                    ),
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.TOPVIEW.value,
                        bucket=mp4_s3_bucket[agent_index],
                        s3_prefix=mp4_s3_object_prefix[agent_index],
                        region_name=aws_region,
                        local_path=CAMERA_TOPVIEW_LOCAL_PATH_FORMAT.format(agent_name),
                    ),
                ]
            )

        run_phase_subject = RunPhaseSubject()
        agent_list.append(
            create_rollout_agent(
                agent_config,
                EvalMetrics(agent_name, metrics_s3_config, args.is_continuous),
                run_phase_subject,
            )
        )
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())

    # ROS service to indicate all the robomaker markov packages are ready for consumption
    signal_robomaker_markov_package_ready()

    PhaseObserver("/agent/training_phase", run_phase_subject)
    enable_domain_randomization = utils.str2bool(
        rospy.get_param("ENABLE_DOMAIN_RANDOMIZATION", False)
    )

    sm_hyperparams_dict = {}

    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service("/gazebo/pause_physics_dr")
    rospy.wait_for_service("/gazebo/unpause_physics_dr")
    pause_physics = ServiceProxyWrapper("/gazebo/pause_physics_dr", Empty)
    unpause_physics = ServiceProxyWrapper("/gazebo/unpause_physics_dr", Empty)

    graph_manager, _ = get_graph_manager(
        hp_dict=sm_hyperparams_dict,
        agent_list=agent_list,
        run_phase_subject=run_phase_subject,
        enable_domain_randomization=enable_domain_randomization,
        done_condition=done_condition,
        pause_physics=pause_physics,
        unpause_physics=unpause_physics,
    )

    ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

    graph_manager.data_store = S3BotoDataStore(
        params=ds_params_instance, graph_manager=graph_manager, ignore_lock=True
    )
    graph_manager.env_params.seed = 0

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.local_model_directory

    evaluation_worker(
        graph_manager=graph_manager,
        number_of_trials=args.number_of_trials,
        task_parameters=task_parameters,
        simtrace_video_s3_writers=simtrace_video_s3_writers,
        is_continuous=args.is_continuous,
        park_positions=park_positions,
        race_type=args.race_type,
        pause_physics=pause_physics,
        unpause_physics=unpause_physics,
    )


if __name__ == "__main__":
    try:
        rospy.init_node("rl_coach", anonymous=True)
        main()
    except ValueError as err:
        if utils.is_user_error(err):
            log_and_exit(
                "User modified model/model_metadata: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
        else:
            log_and_exit(
                "Eval worker value error: {}".format(err),
                SIMAPP_SIMULATION_WORKER_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )
    except GenericRolloutError as ex:
        ex.log_except_and_exit()
    except GenericRolloutException as ex:
        ex.log_except_and_exit()
    except Exception as ex:
        log_and_exit(
            "Eval worker error: {}".format(ex),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )
