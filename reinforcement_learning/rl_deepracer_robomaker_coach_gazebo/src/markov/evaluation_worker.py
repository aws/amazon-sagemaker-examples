'''This module is responsible for launching evaluation jobs'''
import argparse
import json
import logging
import os
import rospy

from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.data_stores.data_store import SyncFiles
from rl_coach.rollout_worker import wait_for_checkpoint

from markov import utils, utils_parse_model_metadata
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import create_rollout_agent, create_obstacles_agent, create_bot_cars_agent
from markov.defaults import reward_function, DEFAULT_MAIN_CAMERA, DEFAULT_SUB_CAMERA
from markov.environments.constants import VELOCITY_TOPICS, STEERING_TOPICS, LINK_NAMES
from markov.metrics.s3_metrics import EvalMetrics
from markov.metrics.constants import MetricsS3Keys
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.s3_client import SageS3Client
from markov.sagemaker_graph_manager import get_graph_manager
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.camera_utils import configure_camera
from markov.utils_parse_model_metadata import parse_model_metadata

from std_srvs.srv import Empty, EmptyRequest

logger = utils.Logger(__name__, logging.INFO).get_logger()

EVALUATION_SIMTRACE_DATA_S3_OBJECT_KEY = "sim_inference_logs/EvaluationSimTraceData.csv"

CUSTOM_FILES_PATH = "./custom_files"
if not os.path.exists(CUSTOM_FILES_PATH):
    os.makedirs(CUSTOM_FILES_PATH)

def evaluation_worker(graph_manager, data_store, number_of_trials, task_parameters):
    checkpoint_dir = task_parameters.checkpoint_restore_path
    wait_for_checkpoint(checkpoint_dir, data_store)
    # Make the clients that will allow us to pause and unpause the physics
    rospy.wait_for_service('/gazebo/pause_physics')
    rospy.wait_for_service('/gazebo/unpause_physics')
    pause_physics = ServiceProxyWrapper('/gazebo/pause_physics', Empty)
    unpause_physics = ServiceProxyWrapper('/gazebo/unpause_physics', Empty)
    graph_manager.create_graph(task_parameters=task_parameters, stop_physics=pause_physics,
                               start_physics=unpause_physics, empty_service_call=EmptyRequest)

    # Instantiate Cameras
    configure_camera()

    unpause_physics(EmptyRequest())
    graph_manager.reset_internal_state(True)
    for _ in range(number_of_trials):
        graph_manager.evaluate(EnvironmentSteps(1))

    # Close the down the job
    utils.cancel_simulation_job(os.environ.get('AWS_ROBOMAKER_SIMULATION_JOB_ARN'),
                                rospy.get_param('AWS_REGION'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preset',
                        help="(string) Name of a preset to run \
                             (class name from the 'presets' directory.)",
                        type=str,
                        required=False)
    parser.add_argument('--s3_bucket',
                        help='(string) S3 bucket',
                        type=str,
                        default=rospy.get_param("MODEL_S3_BUCKET", "gsaur-test"))
    parser.add_argument('--s3_prefix',
                        help='(string) S3 prefix',
                        type=str,
                        default=rospy.get_param("MODEL_S3_PREFIX", "sagemaker"))
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

    args = parser.parse_args()
    logger.info("S3 bucket: %s \n S3 prefix: %s", args.s3_bucket, args.s3_prefix)

    s3_client = SageS3Client(bucket=args.s3_bucket,
                             s3_prefix=args.s3_prefix,
                             aws_region=args.aws_region)

    # Load the model metadata
    model_metadata_local_path = os.path.join(CUSTOM_FILES_PATH, 'model_metadata.json')
    utils.load_model_metadata(s3_client,
                              os.path.normpath("%s/model/model_metadata.json" % args.s3_prefix),
                              model_metadata_local_path)
    # Handle backward compatibility
    _, _, version = parse_model_metadata(model_metadata_local_path)
    if float(version) < float(utils.SIMAPP_VERSION) and \
    not utils.has_current_ckpnt_name(args.s3_bucket, args.s3_prefix, args.aws_region):
        utils.make_compatible(args.s3_bucket, args.s3_prefix, args.aws_region, SyncFiles.TRAINER_READY.value)
    # Download hyperparameters from SageMaker
    hyperparameters_file_success = False
    hyperparams_s3_key = os.path.normpath(args.s3_prefix + "/ip/hyperparameters.json")
    hyperparameters_file_success = s3_client.download_file(s3_key=hyperparams_s3_key,
                                                           local_path="hyperparameters.json")
    sm_hyperparams_dict = {}
    if hyperparameters_file_success:
        logger.info("Received Sagemaker hyperparameters successfully!")
        with open("hyperparameters.json") as file:
            sm_hyperparams_dict = json.load(file)
    else:
        logger.info("SageMaker hyperparameters not found.")

    #! TODO each agent should have own config
    _, _, version = utils_parse_model_metadata.parse_model_metadata(model_metadata_local_path)
    agent_config = {'model_metadata': model_metadata_local_path,
                    'car_ctrl_cnfig': {ConfigParams.LINK_NAME_LIST.value: LINK_NAMES,
                                       ConfigParams.VELOCITY_LIST.value : VELOCITY_TOPICS,
                                       ConfigParams.STEERING_LIST.value : STEERING_TOPICS,
                                       ConfigParams.CHANGE_START.value : utils.str2bool(rospy.get_param('CHANGE_START_POSITION', False)),
                                       ConfigParams.ALT_DIR.value : utils.str2bool(rospy.get_param('ALTERNATE_DRIVING_DIRECTION', False)),
                                       ConfigParams.ACTION_SPACE_PATH.value : 'custom_files/model_metadata.json',
                                       ConfigParams.REWARD.value : reward_function,
                                       ConfigParams.AGENT_NAME.value : 'racecar',
                                       ConfigParams.VERSION.value : version}}

    #! TODO each agent should have own s3 bucket
    metrics_s3_config = {MetricsS3Keys.METRICS_BUCKET.value: rospy.get_param('METRICS_S3_BUCKET'),
                         MetricsS3Keys.METRICS_KEY.value:  rospy.get_param('METRICS_S3_OBJECT_KEY'),
                         MetricsS3Keys.REGION.value: rospy.get_param('AWS_REGION'),
                         MetricsS3Keys.STEP_BUCKET.value: rospy.get_param('MODEL_S3_BUCKET'),
                         MetricsS3Keys.STEP_KEY.value: os.path.join(rospy.get_param('MODEL_S3_PREFIX'),
                                                                    EVALUATION_SIMTRACE_DATA_S3_OBJECT_KEY)}

    agent_list = list()
    agent_list.append(create_rollout_agent(agent_config, EvalMetrics(metrics_s3_config)))
    agent_list.append(create_obstacles_agent())
    agent_list.append(create_bot_cars_agent())

    graph_manager, _ = get_graph_manager(sm_hyperparams_dict, agent_list)

    ds_params_instance = S3BotoDataStoreParameters(aws_region=args.aws_region,
                                                   bucket_name=args.s3_bucket,
                                                   checkpoint_dir=args.local_model_directory,
                                                   s3_folder=args.s3_prefix)

    data_store = S3BotoDataStore(ds_params_instance)
    data_store.graph_manager = graph_manager
    graph_manager.data_store = data_store
    graph_manager.env_params.seed = 0

    task_parameters = TaskParameters()
    task_parameters.checkpoint_restore_path = args.local_model_directory

    evaluation_worker(
        graph_manager=graph_manager,
        data_store=data_store,
        number_of_trials=args.number_of_trials,
        task_parameters=task_parameters,
    )

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
    except Exception as ex:
        utils.log_and_exit("Eval worker error: {}".format(ex),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)
