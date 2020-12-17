'''This module is responsible for launching evaluation jobs'''
import argparse
import json
import logging
import os
import time
import rospy
import pickle

from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.data_stores.data_store import SyncFiles
from markov import utils
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.constants import (SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500)
from markov.constants import SIMAPP_VERSION_2, DEFAULT_PARK_POSITION, ROLLOUT_WORKER_PROFILER_PATH
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import create_rollout_agent, create_obstacles_agent, create_bot_cars_agent
from markov.agents.utils import RunPhaseSubject
from markov.defaults import reward_function
from markov.log_handler.deepracer_exceptions import GenericRolloutError, GenericRolloutException
from markov.environments.constants import VELOCITY_TOPICS, STEERING_TOPICS, LINK_NAMES
from markov.metrics.s3_metrics import EvalMetrics
from markov.metrics.iteration_data import IterationData
from markov.metrics.constants import MetricsS3Keys
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from markov.rollout_utils import (PhaseObserver, signal_robomaker_markov_package_ready,
                                  configure_environment_randomizer, get_robomaker_profiler_env)
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.camera_utils import configure_camera
from markov.track_geom.track_data import TrackData
from markov.track_geom.utils import get_start_positions
from markov.reset.constants import AgentInfo
from markov.s3.constants import (MODEL_METADATA_LOCAL_PATH_FORMAT,
                                 MODEL_METADATA_S3_POSTFIX,
                                 SIMTRACE_EVAL_LOCAL_PATH_FORMAT,
                                 CAMERA_PIP_MP4_LOCAL_PATH_FORMAT,
                                 CAMERA_45DEGREE_LOCAL_PATH_FORMAT,
                                 CAMERA_TOPVIEW_LOCAL_PATH_FORMAT,
                                 SimtraceVideoNames)
from markov.s3.files.model_metadata import ModelMetadata
from markov.s3.files.simtrace_video import SimtraceVideo
from markov.s3.files.checkpoint import Checkpoint
from markov.s3.utils import get_s3_key

from std_srvs.srv import Empty, EmptyRequest

logger = Logger(__name__, logging.INFO).get_logger()

MIN_RESET_COUNT = 10000 #TODO: change when console passes float("inf")

IS_PROFILER_ON, PROFILER_S3_BUCKET, PROFILER_S3_PREFIX = get_robomaker_profiler_env()

def tournament_worker(graph_manager, number_of_trials, task_parameters, simtrace_video_s3_writers, is_continuous,
                      park_positions):
    """ Tournament worker function

    Arguments:
        graph_manager(MultiAgentGraphManager): Graph manager of multiagent graph manager
        number_of_trials(int): Number of trails you want to run the evaluation
        task_parameters(TaskParameters): Information of the checkpoint, gpu/cpu,
            framework etc of rlcoach
        simtrace_video_s3_writers(list): Information to upload to the S3 bucket all the simtrace and mp4
        is_continuous(bool): The termination condition for the car
        park_positions(list of tuple): list of (x, y) for cars to park at
    """
    # Collect profiler information only IS_PROFILER_ON is true
    with utils.Profiler(s3_bucket=PROFILER_S3_BUCKET, s3_prefix=PROFILER_S3_PREFIX,
                        output_local_path=ROLLOUT_WORKER_PROFILER_PATH, enable_profiling=IS_PROFILER_ON):
        checkpoint_dirs = list()
        agent_names = list()
        subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic = list(), list()
        subscribe_to_save_mp4, unsubscribe_from_save_mp4 = list(), list()
        for agent_param in graph_manager.agents_params:
            _checkpoint_dir = os.path.join(task_parameters.checkpoint_restore_path, agent_param.name)
            agent_names.append(agent_param.name)
            checkpoint_dirs.append(_checkpoint_dir)
            racecar_name = 'racecar' if len(agent_param.name.split("_")) == 1 \
                                     else "racecar_{}".format(agent_param.name.split("_")[1])
            subscribe_to_save_mp4_topic.append("/{}/save_mp4/subscribe_to_save_mp4".format(racecar_name))
            unsubscribe_from_save_mp4_topic.append("/{}/save_mp4/unsubscribe_from_save_mp4".format(racecar_name))
        graph_manager.data_store.wait_for_checkpoints()
        graph_manager.data_store.modify_checkpoint_variables()

        # Make the clients that will allow us to pause and unpause the physics
        rospy.wait_for_service('/gazebo/pause_physics_dr')
        rospy.wait_for_service('/gazebo/unpause_physics_dr')
        pause_physics = ServiceProxyWrapper('/gazebo/pause_physics_dr', Empty)
        unpause_physics = ServiceProxyWrapper('/gazebo/unpause_physics_dr', Empty)

        for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
            rospy.wait_for_service(mp4_sub)
            rospy.wait_for_service(mp4_unsub)
        for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
            subscribe_to_save_mp4.append(ServiceProxyWrapper(mp4_sub, Empty))
            unsubscribe_from_save_mp4.append(ServiceProxyWrapper(mp4_unsub, Empty))

        graph_manager.create_graph(task_parameters=task_parameters, stop_physics=pause_physics,
                                   start_physics=unpause_physics, empty_service_call=EmptyRequest)
        logger.info("Graph manager successfully created the graph: Unpausing physics")
        unpause_physics(EmptyRequest())

        is_save_mp4_enabled = rospy.get_param('MP4_S3_BUCKET', None)
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
                unsubscribe_mp4(EmptyRequest())
        # upload simtrace and mp4 into s3 bucket
        for s3_writer in simtrace_video_s3_writers:
            s3_writer.persist(utils.get_s3_kms_extra_args())
        time.sleep(1)
        pause_physics(EmptyRequest())

    # tournament_worker: DO NOT cancel RoboMaker job
    # Close the down the job
    # utils.cancel_simulation_job(os.environ.get('AWS_ROBOMAKER_SIMULATION_JOB_ARN'),
    #                            rospy.get_param('AWS_REGION'))


# tournament_worker: The order of list will be the order that ros node gets killed.
ROS_NODE_PREFIX_LIST_TO_TERMINATE = [
    # '/save_to_mp4',
    # '/kinesis_video_camera_node',
    # '/car_reset_node',
    # '/visualization_node',
    '/tournament_race_node',
    # '/racecar_0/controller_manager',
    # '/racecar_0/robot_state_publisher',
    # '/racecar_1/controller_manager',
    # '/racecar_1/robot_state_publisher',
    # '/rl_coach',
    # '/robomaker/srv',
    # '/rosout',
    # '/rqt_gui_cpp_node',
    # '/rviz',
    # '/gazebo'
]
ROS_NODE_PREFIX_TO_INDEX_MAP = {ros_node_prefix: idx
                                for idx, ros_node_prefix in enumerate(ROS_NODE_PREFIX_LIST_TO_TERMINATE)}


# tournament_worker: terminate tournament_race_node
# this will cause `tournament_node` to restart the RoboMaker job for next race.
def terminate_tournament_race():
    # Terminate tournament_race_node
    node_names = os.popen("rosnode list").readlines()
    for i in range(len(node_names)):
        node_names[i] = node_names[i].replace("\n", "")
    logger.info("ROS nodes running: {}".format(node_names))

    # Map the termination index to actual node name
    terminate_index_to_node_map = {}
    for node_name in node_names:
        for node_prefix_to_terminate in ROS_NODE_PREFIX_TO_INDEX_MAP:
            if node_name.startswith(node_prefix_to_terminate):
                terminate_index_to_node_map[ROS_NODE_PREFIX_TO_INDEX_MAP[node_prefix_to_terminate]] = node_name
                break
    # Sort by key and kill node in order.
    for index in sorted(terminate_index_to_node_map):
        logger.info("Killing ROS node ({})...".format(terminate_index_to_node_map[index]))
        os.system("rosnode kill {}".format(terminate_index_to_node_map[index]))


# tournament_worker: write race report before exiting the node.
def write_race_report(graph_manager,
                      model_s3_bucket_map, model_s3_prefix_map,
                      metrics_s3_bucket_map, metrics_s3_key_map,
                      simtrace_s3_bucket_map, simtrace_s3_prefix_map,
                      mp4_s3_bucket_map, mp4_s3_prefix_map,
                      display_names):
    env = graph_manager.environments[0]
    agents_info_map = env.agents_info_map

    best_racecar_name = None
    best_agent_lap = None
    best_agent_progress = None
    racecar_names = sorted(agents_info_map.keys())
    for racecar_name, agent_info in agents_info_map.items():
        if best_racecar_name is None:
            best_racecar_name = racecar_name
            best_agent_lap = agent_info[AgentInfo.LAP_COUNT.value]
            best_agent_progress = agent_info[AgentInfo.CURRENT_PROGRESS.value]
        else:
            cur_agent_lap = agent_info[AgentInfo.LAP_COUNT.value]
            cur_agent_progress = agent_info[AgentInfo.CURRENT_PROGRESS.value]

            if cur_agent_lap > best_agent_lap or \
                    (cur_agent_lap == best_agent_lap and cur_agent_progress > best_agent_progress):
                best_racecar_name = racecar_name
                best_agent_lap = cur_agent_lap
                best_agent_progress = cur_agent_progress

    race_result = {}
    for agent_idx, (racecar_name, display_name) in enumerate(zip(racecar_names, display_names)):
        key = 'racer' + str(agent_idx + 1)
        agent_name = racecar_name.replace('racecar', 'agent')
        race_result[key] = {
            "display_name": display_name,
            "model": {
                "s3_bucket": model_s3_bucket_map[agent_name],
                "s3_prefix": model_s3_prefix_map[agent_name],
            },
            "metric": {
                "s3_bucket": metrics_s3_bucket_map[agent_name],
                "s3_key": metrics_s3_key_map[agent_name]
            },
            "video": {
                "s3_bucket": mp4_s3_bucket_map[agent_name],
                "s3_prefix": mp4_s3_prefix_map[agent_name]
            },
            "simtrace": {
                "s3_bucket": simtrace_s3_bucket_map[agent_name],
                "s3_prefix": simtrace_s3_prefix_map[agent_name]
            }
        }
        if racecar_name == best_racecar_name:
            race_result["winner"] = display_name
    with open('race_report.pkl', 'wb') as f:
        pickle.dump(race_result, f, protocol=2)


def main():
    """ Main function for tournament worker """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run \
                             (class name from the 'presets' directory.)",
                        type=str,
                        required=False)
    parser.add_argument('--s3_bucket',
                        help='list(string) S3 bucket',
                        type=str,
                        nargs='+',
                        default=rospy.get_param("MODEL_S3_BUCKET", ["gsaur-test"]))
    parser.add_argument('--s3_prefix',
                        help='list(string) S3 prefix',
                        type=str,
                        nargs='+',
                        default=rospy.get_param("MODEL_S3_PREFIX", ["sagemaker"]))
    parser.add_argument('--s3_endpoint_url',
                        help='(string) S3 endpoint URL',
                        type=str,
                        default=rospy.get_param("S3_ENDPOINT_URL", None))                             
    parser.add_argument('--aws_region',
                        help='(string) AWS region',
                        type=str,
                        default=rospy.get_param("AWS_REGION", "us-east-1"))
    parser.add_argument('--number_of_trials',
                        help='(integer) Number of trials',
                        type=int,
                        default=int(rospy.get_param("NUMBER_OF_TRIALS", 10)))
    parser.add_argument('-c', '--local_model_directory',
                        help='(string) Path to a folder containing a checkpoint \
                             to restore the model from.',
                        type=str,
                        default='./checkpoint')
    parser.add_argument('--number_of_resets',
                        help='(integer) Number of resets',
                        type=int,
                        default=int(rospy.get_param("NUMBER_OF_RESETS", 0)))
    parser.add_argument('--penalty_seconds',
                        help='(float) penalty second',
                        type=float,
                        default=float(rospy.get_param("PENALTY_SECONDS", 2.0)))
    parser.add_argument('--job_type',
                        help='(string) job type',
                        type=str,
                        default=rospy.get_param("JOB_TYPE", "EVALUATION"))
    parser.add_argument('--is_continuous',
                        help='(boolean) is continous after lap completion',
                        type=bool,
                        default=utils.str2bool(rospy.get_param("IS_CONTINUOUS", False)))
    parser.add_argument('--race_type',
                        help='(string) Race type',
                        type=str,
                        default=rospy.get_param("RACE_TYPE", "TIME_TRIAL"))
    parser.add_argument('--off_track_penalty',
                        help='(float) off track penalty second',
                        type=float,
                        default=float(rospy.get_param("OFF_TRACK_PENALTY", 2.0)))
    parser.add_argument('--collision_penalty',
                        help='(float) collision penalty second',
                        type=float,
                        default=float(rospy.get_param("COLLISION_PENALTY", 5.0)))

    args = parser.parse_args()
    arg_s3_bucket = args.s3_bucket
    arg_s3_prefix = args.s3_prefix
    logger.info("S3 bucket: %s", args.s3_bucket)
    logger.info("S3 prefix: %s", args.s3_prefix) 
    logger.info("S3 endpoint URL: %s" % args.s3_endpoint_url)

    # tournament_worker: names to be displayed in MP4.
    # This is racer alias in tournament worker case.
    display_names = utils.get_video_display_name()

    metrics_s3_buckets = rospy.get_param('METRICS_S3_BUCKET')
    metrics_s3_object_keys = rospy.get_param('METRICS_S3_OBJECT_KEY')

    arg_s3_bucket, arg_s3_prefix = utils.force_list(arg_s3_bucket), utils.force_list(arg_s3_prefix)
    metrics_s3_buckets = utils.force_list(metrics_s3_buckets)
    metrics_s3_object_keys = utils.force_list(metrics_s3_object_keys)

    validate_list = [arg_s3_bucket, arg_s3_prefix, metrics_s3_buckets, metrics_s3_object_keys]

    simtrace_s3_bucket = rospy.get_param('SIMTRACE_S3_BUCKET', None)
    mp4_s3_bucket = rospy.get_param('MP4_S3_BUCKET', None)
    if simtrace_s3_bucket:
        simtrace_s3_object_prefix = rospy.get_param('SIMTRACE_S3_PREFIX')
        simtrace_s3_bucket = utils.force_list(simtrace_s3_bucket)
        simtrace_s3_object_prefix = utils.force_list(simtrace_s3_object_prefix)
        validate_list.extend([simtrace_s3_bucket, simtrace_s3_object_prefix])
    if mp4_s3_bucket:
        mp4_s3_object_prefix = rospy.get_param('MP4_S3_OBJECT_PREFIX')
        mp4_s3_bucket = utils.force_list(mp4_s3_bucket)
        mp4_s3_object_prefix = utils.force_list(mp4_s3_object_prefix)
        validate_list.extend([mp4_s3_bucket, mp4_s3_object_prefix])

    if not all([lambda x: len(x) == len(validate_list[0]), validate_list]):
        log_and_exit("Tournament worker error: Incorrect arguments passed: {}"
                         .format(validate_list),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
    if args.number_of_resets != 0 and args.number_of_resets < MIN_RESET_COUNT:
        raise GenericRolloutException("number of resets is less than {}".format(MIN_RESET_COUNT))

    # Instantiate Cameras
    if len(arg_s3_bucket) == 1:
        configure_camera(namespaces=['racecar'])
    else:
        configure_camera(namespaces=[
            'racecar_{}'.format(str(agent_index)) for agent_index in range(len(arg_s3_bucket))])

    agent_list = list()
    s3_bucket_dict = dict()
    s3_prefix_dict = dict()
    checkpoint_dict = dict()
    start_positions = get_start_positions(len(arg_s3_bucket))
    done_condition = utils.str_to_done_condition(rospy.get_param("DONE_CONDITION", any))
    park_positions = utils.pos_2d_str_to_list(rospy.get_param("PARK_POSITIONS", []))
    # if not pass in park positions for all done condition case, use default
    if not park_positions:
        park_positions = [DEFAULT_PARK_POSITION for _ in arg_s3_bucket]

    # tournament_worker: list of required S3 locations
    simtrace_s3_bucket_dict = dict()
    simtrace_s3_prefix_dict = dict()
    metrics_s3_bucket_dict = dict()
    metrics_s3_obect_key_dict = dict()
    mp4_s3_bucket_dict = dict()
    mp4_s3_object_prefix_dict = dict()
    simtrace_video_s3_writers = []

    for agent_index, s3_bucket_val in enumerate(arg_s3_bucket):
        agent_name = 'agent' if len(arg_s3_bucket) == 1 else 'agent_{}'.format(str(agent_index))
        racecar_name = 'racecar' if len(arg_s3_bucket) == 1 else 'racecar_{}'.format(str(agent_index))
        s3_bucket_dict[agent_name] = arg_s3_bucket[agent_index]
        s3_prefix_dict[agent_name] = arg_s3_prefix[agent_index]

        # tournament_worker: remap key with agent_name instead of agent_index for list of S3 locations.
        simtrace_s3_bucket_dict[agent_name] = simtrace_s3_bucket[agent_index]
        simtrace_s3_prefix_dict[agent_name] = simtrace_s3_object_prefix[agent_index]
        metrics_s3_bucket_dict[agent_name] = metrics_s3_buckets[agent_index]
        metrics_s3_obect_key_dict[agent_name] = metrics_s3_object_keys[agent_index]
        mp4_s3_bucket_dict[agent_name] = mp4_s3_bucket[agent_index]
        mp4_s3_object_prefix_dict[agent_name] = mp4_s3_object_prefix[agent_index]

        # download model metadata
        model_metadata = ModelMetadata(bucket=arg_s3_bucket[agent_index],
                                       s3_key=get_s3_key(arg_s3_prefix[agent_index], MODEL_METADATA_S3_POSTFIX),
                                       region_name=args.aws_region,
                                       s3_endpoint_url=args.s3_endpoint_url,
                                       local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format(agent_name))
        _, _, version = model_metadata.get_model_metadata_info()

        # checkpoint s3 instance
        checkpoint = Checkpoint(bucket=arg_s3_bucket[agent_index],
                                s3_prefix=arg_s3_prefix[agent_index],
                                region_name=args.aws_region,
                                s3_endpoint_url=args.s3_endpoint_url,
                                agent_name=agent_name,
                                checkpoint_dir=args.local_model_directory)
        # make coach checkpoint compatible
        if version < SIMAPP_VERSION_2 and not checkpoint.rl_coach_checkpoint.is_compatible():
            checkpoint.rl_coach_checkpoint.make_compatible(checkpoint.syncfile_ready)
        # get best model checkpoint string
        model_checkpoint_name = checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint()
        # Select the best checkpoint model by uploading rl coach .coach_checkpoint file
        checkpoint.rl_coach_checkpoint.update(
            model_checkpoint_name=model_checkpoint_name,
            s3_kms_extra_args=utils.get_s3_kms_extra_args())

        checkpoint_dict[agent_name] = checkpoint

        agent_config = {
            'model_metadata': model_metadata,
            ConfigParams.CAR_CTRL_CONFIG.value: {
                ConfigParams.LINK_NAME_LIST.value: [
                    link_name.replace('racecar', racecar_name) for link_name in LINK_NAMES],
                ConfigParams.VELOCITY_LIST.value: [
                    velocity_topic.replace('racecar', racecar_name) for velocity_topic in VELOCITY_TOPICS],
                ConfigParams.STEERING_LIST.value: [
                    steering_topic.replace('racecar', racecar_name) for steering_topic in STEERING_TOPICS],
                ConfigParams.CHANGE_START.value: utils.str2bool(rospy.get_param('CHANGE_START_POSITION', False)),
                ConfigParams.ALT_DIR.value: utils.str2bool(rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)),
                ConfigParams.ACTION_SPACE_PATH.value: model_metadata.local_path,
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
                ConfigParams.DONE_CONDITION.value: done_condition}}

        metrics_s3_config = {MetricsS3Keys.METRICS_BUCKET.value: metrics_s3_buckets[agent_index],
                             MetricsS3Keys.METRICS_KEY.value: metrics_s3_object_keys[agent_index],
                             MetricsS3Keys.ENDPOINT_URL.value:  rospy.get_param('S3_ENDPOINT_URL', None),
                             # Replaced rospy.get_param('AWS_REGION') to be equal to the argument being passed
                             # or default argument set
                             MetricsS3Keys.REGION.value: args.aws_region}
        aws_region = rospy.get_param('AWS_REGION', args.aws_region)
        if simtrace_s3_bucket:
            simtrace_video_s3_writers.append(
                SimtraceVideo(upload_type=SimtraceVideoNames.SIMTRACE_EVAL.value,
                              bucket=simtrace_s3_bucket[agent_index],
                              s3_prefix=simtrace_s3_object_prefix[agent_index],
                              region_name=aws_region,
                              s3_endpoint_url=args.s3_endpoint_url,
                              local_path=SIMTRACE_EVAL_LOCAL_PATH_FORMAT.format(agent_name)))
        if mp4_s3_bucket:
            simtrace_video_s3_writers.extend([
                SimtraceVideo(upload_type=SimtraceVideoNames.PIP.value,
                              bucket=mp4_s3_bucket[agent_index],
                              s3_prefix=mp4_s3_object_prefix[agent_index],
                              region_name=aws_region,
                              s3_endpoint_url=args.s3_endpoint_url,
                              local_path=CAMERA_PIP_MP4_LOCAL_PATH_FORMAT.format(agent_name)),
                SimtraceVideo(upload_type=SimtraceVideoNames.DEGREE45.value,
                              bucket=mp4_s3_bucket[agent_index],
                              s3_prefix=mp4_s3_object_prefix[agent_index],
                              region_name=aws_region,
                              s3_endpoint_url=args.s3_endpoint_url,
                              local_path=CAMERA_45DEGREE_LOCAL_PATH_FORMAT.format(agent_name)),
                SimtraceVideo(upload_type=SimtraceVideoNames.TOPVIEW.value,
                              bucket=mp4_s3_bucket[agent_index],
                              s3_prefix=mp4_s3_object_prefix[agent_index],
                              region_name=aws_region,
                              s3_endpoint_url=args.s3_endpoint_url,
                              local_path=CAMERA_TOPVIEW_LOCAL_PATH_FORMAT.format(agent_name))])

        run_phase_subject = RunPhaseSubject()
        agent_list.append(create_rollout_agent(agent_config, EvalMetrics(agent_name, metrics_s3_config,
                                                                         args.is_continuous),
                                               run_phase_subject))
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())

    # ROS service to indicate all the robomaker markov packages are ready for consumption
    signal_robomaker_markov_package_ready()

    PhaseObserver('/agent/training_phase', run_phase_subject)
    enable_domain_randomization = utils.str2bool(rospy.get_param('ENABLE_DOMAIN_RANDOMIZATION', False))

    sm_hyperparams_dict = {}
    graph_manager, _ = get_graph_manager(hp_dict=sm_hyperparams_dict, agent_list=agent_list,
                                         run_phase_subject=run_phase_subject,
                                         enable_domain_randomization=enable_domain_randomization,
                                         done_condition=done_condition)

    ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

    graph_manager.data_store = S3BotoDataStore(params=ds_params_instance,
                                               graph_manager=graph_manager,
                                               ignore_lock=True)
    graph_manager.env_params.seed = 0

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.local_model_directory

    tournament_worker(
        graph_manager=graph_manager,
        number_of_trials=args.number_of_trials,
        task_parameters=task_parameters,
        simtrace_video_s3_writers=simtrace_video_s3_writers,
        is_continuous=args.is_continuous,
        park_positions=park_positions
    )

    # tournament_worker: write race report to local file.
    write_race_report(graph_manager,
                      model_s3_bucket_map=s3_bucket_dict, model_s3_prefix_map=s3_prefix_dict,
                      metrics_s3_bucket_map=metrics_s3_bucket_dict, metrics_s3_key_map=metrics_s3_obect_key_dict,
                      simtrace_s3_bucket_map=simtrace_s3_bucket_dict, simtrace_s3_prefix_map=simtrace_s3_prefix_dict,
                      mp4_s3_bucket_map=mp4_s3_bucket_dict, mp4_s3_prefix_map=mp4_s3_object_prefix_dict,
                      display_names=display_names)

    # tournament_worker: terminate tournament_race_node.
    terminate_tournament_race()


if __name__ == '__main__':
    try:
        rospy.init_node('rl_coach', anonymous=True)
        main()
    except ValueError as err:
        if utils.is_user_error(err):
            log_and_exit("User modified model/model_metadata: {}".format(err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
        else:
            log_and_exit("Tournament worker value error: {}"
                             .format(err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
    except GenericRolloutError as ex:
        ex.log_except_and_exit()
    except GenericRolloutException as ex:
        ex.log_except_and_exit()
    except Exception as ex:
        log_and_exit("Tournament worker error: {}"
                         .format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION,
                     SIMAPP_EVENT_ERROR_CODE_500)
