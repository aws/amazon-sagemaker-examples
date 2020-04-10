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
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import create_rollout_agent, create_obstacles_agent, create_bot_cars_agent
from markov.agents.utils import RunPhaseSubject
from markov.defaults import reward_function
from markov.deepracer_exceptions import GenericRolloutError, GenericRolloutException
from markov.environments.constants import VELOCITY_TOPICS, STEERING_TOPICS, LINK_NAMES
from markov.metrics.s3_metrics import EvalMetrics
from markov.metrics.s3_writer import S3Writer
from markov.metrics.iteration_data import IterationData
from markov.metrics.constants import MetricsS3Keys, IterationDataLocalFileNames, ITERATION_DATA_LOCAL_FILE_PATH
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.rollout_utils import PhaseObserver, signal_robomaker_markov_package_ready
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.camera_utils import configure_camera
from markov.utils_parse_model_metadata import parse_model_metadata
from markov.checkpoint_utils import TEMP_RENAME_FOLDER, wait_for_checkpoints, modify_checkpoint_variables
from markov.reset.constants import AgentInfo

from std_srvs.srv import Empty, EmptyRequest

logger = utils.Logger(__name__, logging.INFO).get_logger()

EVALUATION_SIMTRACE_DATA_S3_OBJECT_KEY = "sim_inference_logs/EvaluationSimTraceData.csv"
MIN_RESET_COUNT = 10000 #TODO: change when console passes float("inf")

CUSTOM_FILES_PATH = "./custom_files"
if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)

if not os.path.exists(TEMP_RENAME_FOLDER):
    os.makedirs(TEMP_RENAME_FOLDER)


def tournament_worker(graph_manager, number_of_trials, task_parameters, s3_writers, is_continuous):
    """ Tournament worker function

    Arguments:
        graph_manager {[MultiAgentGraphManager]} -- [Graph manager of multiagent graph manager]
        number_of_trials {[int]} -- [Number of trails you want to run the evaluation]
        task_parameters {[TaskParameters]} -- [Information of the checkpoint, gpu/cpu, framework etc of rlcoach]
        s3_writers {[S3Writer]} -- [Information to upload to the S3 bucket all the simtrace and mp4]
        is_continuous {bool} -- [The termination condition for the car]
    """
    checkpoint_dirs = list()
    agent_names = list()
    subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic = list(), list()
    subscribe_to_save_mp4, unsubscribe_from_save_mp4 = list(), list()
    for agent_param in graph_manager.agents_params:
        _checkpoint_dir = task_parameters.checkpoint_restore_path if len(graph_manager.agents_params) == 1 \
            else os.path.join(task_parameters.checkpoint_restore_path, agent_param.name)
        agent_names.append(agent_param.name)
        checkpoint_dirs.append(_checkpoint_dir)
        racecar_name = 'racecar' if len(agent_param.name.split("_")) == 1 \
            else "racecar_{}".format(agent_param.name.split("_")[1])
        subscribe_to_save_mp4_topic.append("/{}/save_mp4/subscribe_to_save_mp4".format(racecar_name))
        unsubscribe_from_save_mp4_topic.append("/{}/save_mp4/unsubscribe_from_save_mp4".format(racecar_name))
    wait_for_checkpoints(checkpoint_dirs, graph_manager.data_store)
    modify_checkpoint_variables(checkpoint_dirs, agent_names)

    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service('/gazebo/pause_physics')
    rospy.wait_for_service('/gazebo/unpause_physics')
    pause_physics = ServiceProxyWrapper('/gazebo/pause_physics', Empty)
    unpause_physics = ServiceProxyWrapper('/gazebo/unpause_physics', Empty)

    for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
        rospy.wait_for_service(mp4_sub)
        rospy.wait_for_service(mp4_unsub)
    for mp4_sub, mp4_unsub in zip(subscribe_to_save_mp4_topic, unsubscribe_from_save_mp4_topic):
        subscribe_to_save_mp4.append(ServiceProxyWrapper(mp4_sub, Empty))
        unsubscribe_from_save_mp4.append(ServiceProxyWrapper(mp4_unsub, Empty))

    graph_manager.create_graph(task_parameters=task_parameters, stop_physics=pause_physics,
                               start_physics=unpause_physics, empty_service_call=EmptyRequest)
    unpause_physics(EmptyRequest())
    graph_manager.reset_internal_state(True)

    is_save_mp4_enabled = rospy.get_param('MP4_S3_BUCKET', None)
    if is_save_mp4_enabled:
        for subscribe_mp4 in subscribe_to_save_mp4:
            subscribe_mp4(EmptyRequest())
    if is_continuous:
        graph_manager.evaluate(EnvironmentSteps(1))
    else:
        for _ in range(number_of_trials):
            graph_manager.evaluate(EnvironmentSteps(1))
    if is_save_mp4_enabled:
        for unsubscribe_mp4 in unsubscribe_from_save_mp4:
            unsubscribe_mp4(EmptyRequest())
    for s3_writer in s3_writers:
        s3_writer.upload_to_s3()
    time.sleep(1)
    pause_physics(EmptyRequest())

    # tournament_worker: DO NOT cancel RoboMaker job
    # Close the down the job
    # utils.cancel_simulation_job(os.environ.get('AWS_ROBOMAKER_SIMULATION_JOB_ARN'),
    #                            rospy.get_param('AWS_REGION'))


# tournament_worker: terminate tournament_race_node
# this will cause `tournament_node` to restart the RoboMaker job for next race.
def terminate_tournament_race():
    # Terminate tournament_race_node
    nodes = os.popen("rosnode list").readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace("\n", "")
    for node in nodes:
        if node.startswith('/tournament_race_node'):
            os.system("rosnode kill {}".format(node))
            break


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
    racecar_names = agents_info_map.keys()
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
            "metrics": {
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
    logger.info("S3 bucket: %s \n S3 prefix: %s", arg_s3_bucket, arg_s3_prefix)

    # tournament_worker: names to be displayed in MP4.
    # This is racer alias in tournament worker case.
    display_names = rospy.get_param('DISPLAY_NAME', "")

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
        utils.log_and_exit("Eval worker error: Incorrect arguments passed: {}".format(validate_list),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
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
    s3_writers = list()

    # tournament_worker: list of required S3 locations
    simtrace_s3_bucket_dict = dict()
    simtrace_s3_prefix_dict = dict()
    metrics_s3_bucket_dict = dict()
    metrics_s3_obect_key_dict = dict()
    mp4_s3_bucket_dict = dict()
    mp4_s3_object_prefix_dict = dict()

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

        s3_client = SageS3Client(bucket=arg_s3_bucket[agent_index],
                                 s3_prefix=arg_s3_prefix[agent_index],
                                 aws_region=args.aws_region)

        # Load the model metadata
        if not os.path.exists(os.path.join(CUSTOM_FILES_PATH, agent_name)):
            os.makedirs(os.path.join(CUSTOM_FILES_PATH, agent_name))
        model_metadata_local_path = os.path.join(os.path.join(CUSTOM_FILES_PATH, agent_name), 'model_metadata.json')
        utils.load_model_metadata(s3_client,
                                  os.path.normpath("%s/model/model_metadata.json" % arg_s3_prefix[agent_index]),
                                  model_metadata_local_path)
        # Handle backward compatibility
        _, _, version = parse_model_metadata(model_metadata_local_path)
        if float(version) < float(utils.SIMAPP_VERSION) and \
        not utils.has_current_ckpnt_name(arg_s3_bucket[agent_index], arg_s3_prefix[agent_index], args.aws_region):
            utils.make_compatible(arg_s3_bucket[agent_index], arg_s3_prefix[agent_index], args.aws_region,
                                  SyncFiles.TRAINER_READY.value)

        # Select the optimal model
        utils.do_model_selection(s3_bucket=arg_s3_bucket[agent_index],
                                s3_prefix=arg_s3_prefix[agent_index],
                                region=args.aws_region)

        # Download hyperparameters from SageMaker
        if not os.path.exists(agent_name):
            os.makedirs(agent_name)
        hyperparameters_file_success = False
        hyperparams_s3_key = os.path.normpath(arg_s3_prefix[agent_index] + "/ip/hyperparameters.json")
        hyperparameters_file_success = s3_client.download_file(s3_key=hyperparams_s3_key,
                                                            local_path=os.path.join(agent_name,
                                                                                    "hyperparameters.json"))
        sm_hyperparams_dict = {}
        if hyperparameters_file_success:
            logger.info("Received Sagemaker hyperparameters successfully!")
            with open(os.path.join(agent_name, "hyperparameters.json")) as file:
                sm_hyperparams_dict = json.load(file)
        else:
            logger.info("SageMaker hyperparameters not found.")

        agent_config = {
            'model_metadata': model_metadata_local_path,
            ConfigParams.CAR_CTRL_CONFIG.value: {
                ConfigParams.LINK_NAME_LIST.value: [
                    link_name.replace('racecar', racecar_name) for link_name in LINK_NAMES],
                ConfigParams.VELOCITY_LIST.value: [
                    velocity_topic.replace('racecar', racecar_name) for velocity_topic in VELOCITY_TOPICS],
                ConfigParams.STEERING_LIST.value: [
                    steering_topic.replace('racecar', racecar_name) for steering_topic in STEERING_TOPICS],
                ConfigParams.CHANGE_START.value: utils.str2bool(rospy.get_param('CHANGE_START_POSITION', False)),
                ConfigParams.ALT_DIR.value: utils.str2bool(rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)),
                ConfigParams.ACTION_SPACE_PATH.value: 'custom_files/' + agent_name + '/model_metadata.json',
                ConfigParams.REWARD.value: reward_function,
                ConfigParams.AGENT_NAME.value: racecar_name,
                ConfigParams.VERSION.value: version,
                ConfigParams.NUMBER_OF_RESETS.value: args.number_of_resets,
                ConfigParams.PENALTY_SECONDS.value: args.penalty_seconds,
                ConfigParams.NUMBER_OF_TRIALS.value: args.number_of_trials,
                ConfigParams.IS_CONTINUOUS.value: args.is_continuous,
                ConfigParams.RACE_TYPE.value: args.race_type,
                ConfigParams.COLLISION_PENALTY.value: args.collision_penalty,
                ConfigParams.OFF_TRACK_PENALTY.value: args.off_track_penalty}}

        metrics_s3_config = {MetricsS3Keys.METRICS_BUCKET.value: metrics_s3_buckets[agent_index],
                             MetricsS3Keys.METRICS_KEY.value: metrics_s3_object_keys[agent_index],
                             # Replaced rospy.get_param('AWS_REGION') to be equal to the argument being passed
                             # or default argument set
                             MetricsS3Keys.REGION.value: args.aws_region,
                             # Replaced rospy.get_param('MODEL_S3_BUCKET') to be equal to the argument being passed
                             # or default argument set
                             MetricsS3Keys.STEP_BUCKET.value: arg_s3_bucket[agent_index],
                             # Replaced rospy.get_param('MODEL_S3_PREFIX') to be equal to the argument being passed
                             # or default argument set
                             MetricsS3Keys.STEP_KEY.value: os.path.join(arg_s3_prefix[agent_index],
                                                                        EVALUATION_SIMTRACE_DATA_S3_OBJECT_KEY)}
        aws_region = rospy.get_param('AWS_REGION', args.aws_region)
        s3_writer_job_info = []
        if simtrace_s3_bucket:
            s3_writer_job_info.append(
                IterationData('simtrace', simtrace_s3_bucket[agent_index], simtrace_s3_object_prefix[agent_index],
                              aws_region,
                              os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, agent_name,
                                           IterationDataLocalFileNames.SIM_TRACE_EVALUATION_LOCAL_FILE.value)))
        if mp4_s3_bucket:
            s3_writer_job_info.extend([
                IterationData('pip', mp4_s3_bucket[agent_index], mp4_s3_object_prefix[agent_index], aws_region,
                              os.path.join(
                                  ITERATION_DATA_LOCAL_FILE_PATH, agent_name,
                                  IterationDataLocalFileNames.CAMERA_PIP_MP4_VALIDATION_LOCAL_PATH.value)),
                IterationData('45degree', mp4_s3_bucket[agent_index], mp4_s3_object_prefix[agent_index], aws_region,
                              os.path.join(
                                  ITERATION_DATA_LOCAL_FILE_PATH, agent_name,
                                  IterationDataLocalFileNames.CAMERA_45DEGREE_MP4_VALIDATION_LOCAL_PATH.value)),
                IterationData('topview', mp4_s3_bucket[agent_index], mp4_s3_object_prefix[agent_index], aws_region,
                              os.path.join(
                                  ITERATION_DATA_LOCAL_FILE_PATH, agent_name,
                                  IterationDataLocalFileNames.CAMERA_TOPVIEW_MP4_VALIDATION_LOCAL_PATH.value))])

        s3_writers.append(S3Writer(job_info=s3_writer_job_info))
        run_phase_subject = RunPhaseSubject()
        agent_list.append(create_rollout_agent(agent_config, EvalMetrics(agent_name, metrics_s3_config),
                                               run_phase_subject))
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())
    # ROS service to indicate all the robomaker markov packages are ready for consumption
    signal_robomaker_markov_package_ready()

    PhaseObserver('/agent/training_phase', run_phase_subject)

    graph_manager, _ = get_graph_manager(hp_dict=sm_hyperparams_dict, agent_list=agent_list,
                                         run_phase_subject=run_phase_subject)

    ds_params_instance = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                   bucket_names=s3_bucket_dict,
                                                   base_checkpoint_dir=args.local_model_directory,
                                                   s3_folders=s3_prefix_dict)

    graph_manager.data_store = S3BotoDataStore(params=ds_params_instance, graph_manager=graph_manager,
                                               ignore_lock=True)
    graph_manager.env_params.seed = 0

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.local_model_directory

    tournament_worker(
        graph_manager=graph_manager,
        number_of_trials=args.number_of_trials,
        task_parameters=task_parameters,
        s3_writers=s3_writers,
        is_continuous=args.is_continuous
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
        if utils.is_error_bad_ckpnt(err):
            utils.log_and_exit("User modified model: {}".format(err),
                               utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_400)
        else:
            utils.log_and_exit("Eval worker value error: {}".format(err),
                               utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_500)
    except GenericRolloutError as ex:
        ex.log_except_and_exit()
    except GenericRolloutException as ex:
        ex.log_except_and_exit()
    except Exception as ex:
        utils.log_and_exit("Eval worker error: {}".format(ex),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
