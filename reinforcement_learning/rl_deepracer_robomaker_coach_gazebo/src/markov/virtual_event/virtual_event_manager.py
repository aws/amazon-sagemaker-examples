"""This module is responsible for launching virtual event jobs"""
import io
import json
import logging
import math
import os
import shutil
import time
from collections import namedtuple

import boto3
import botocore
import markov.rollout_constants as const
import rospkg
import rospy
import tensorflow as tf
from deepracer_simulation_environment.srv import (
    VirtualEventVideoEditSrv,
    VirtualEventVideoEditSrvRequest,
)
from gazebo_msgs.msg import ModelState
from markov import utils
from markov.agent_ctrl.constants import ConfigParams
from markov.agents.rollout_agent_factory import (
    create_bot_cars_agent,
    create_obstacles_agent,
    create_rollout_agent,
)
from markov.agents.utils import RunPhaseSubject
from markov.architecture.constants import Input
from markov.auth.refreshed_session import refreshed_session
from markov.boto.s3.constants import (
    CAMERA_45DEGREE_LOCAL_PATH_FORMAT,
    CAMERA_PIP_MP4_LOCAL_PATH_FORMAT,
    CAMERA_TOPVIEW_LOCAL_PATH_FORMAT,
    MODEL_METADATA_LOCAL_PATH_FORMAT,
    MODEL_METADATA_S3_POSTFIX,
    S3_RACE_STATUS_FILE_NAME,
    SECTOR_TIME_LOCAL_PATH,
    SECTOR_TIME_S3_POSTFIX,
    SECTOR_X_FORMAT,
    SIMTRACE_EVAL_LOCAL_PATH_FORMAT,
    ModelMetadataKeys,
    RaceStatusKeys,
    SimtraceVideoNames,
)
from markov.boto.s3.files.checkpoint import Checkpoint
from markov.boto.s3.files.model_metadata import ModelMetadata
from markov.boto.s3.files.simtrace_video import SimtraceVideo
from markov.boto.s3.files.virtual_event_best_sector_time import VirtualEventBestSectorTime
from markov.boto.s3.s3_client import S3Client
from markov.boto.s3.utils import get_s3_key
from markov.boto.sqs.sqs_client import SQSClient
from markov.camera_utils import configure_camera
from markov.cameras.camera_manager import CameraManager
from markov.constants import SIMAPP_VERSION_2
from markov.defaults import reward_function
from markov.environments.constants import LINK_NAMES, STEERING_TOPICS, VELOCITY_TOPICS
from markov.gazebo_utils.model_updater import ModelUpdater
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_EVENT_USER_ERROR,
    SIMAPP_SQS_RECEIVE_MESSAGE_EXCEPTION,
    SIMAPP_VIRTUAL_EVENT_RACE_EXCEPTION,
)
from markov.log_handler.deepracer_exceptions import GenericNonFatalException
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.metrics.constants import MetricsS3Keys
from markov.metrics.s3_metrics import EvalMetrics
from markov.rollout_utils import (
    PhaseObserver,
    configure_environment_randomizer,
    get_robomaker_profiler_env,
    signal_robomaker_markov_package_ready,
)
from markov.rospy_wrappers import ServiceProxyWrapper
from markov.s3_boto_data_store import S3BotoDataStore, S3BotoDataStoreParameters
from markov.sagemaker_graph_manager import get_graph_manager
from markov.track_geom.track_data import FiniteDifference, TrackData
from markov.track_geom.utils import euler_to_quaternion, get_hide_positions, get_start_positions
from markov.utils import get_boto_config, get_racecar_idx
from markov.virtual_event.constants import (
    MAX_NUM_OF_SQS_ERROR,
    MAX_NUM_OF_SQS_MESSAGE,
    PAUSE_TIME_BEFORE_START,
    RACER_INFO_JSON_SCHEMA,
    RACER_INFO_OBJECT,
    SENSOR_MODEL_MAP,
    SQS_WAIT_TIME_SEC,
    VIRTUAL_EVENT,
)
from markov.virtual_event.utils import validate_json_input
from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentSteps
from std_srvs.srv import Empty, EmptyRequest

LOG = Logger(__name__, logging.INFO).get_logger()


class VirtualEventManager(object):
    """
    This is the manager that manages the live virtual event.
    """

    def __init__(
        self,
        queue_url,
        aws_region="us-east-1",
        race_duration=180,
        number_of_trials=3,
        number_of_resets=10000,
        penalty_seconds=2.0,
        off_track_penalty=2.0,
        collision_penalty=5.0,
        is_continuous=False,
        race_type="TIME_TRIAL",
        body_shell_type="deepracer",
    ):
        # constructor arguments
        self._body_shell_type = body_shell_type
        self._num_sectors = int(rospy.get_param("NUM_SECTORS", "3"))
        self._queue_url = queue_url
        self._region = aws_region
        self._number_of_trials = number_of_trials
        self._number_of_resets = number_of_resets
        self._penalty_seconds = penalty_seconds
        self._off_track_penalty = off_track_penalty
        self._collision_penalty = collision_penalty
        self._is_continuous = is_continuous
        self._race_type = race_type
        self._is_save_simtrace_enabled = False
        self._is_save_mp4_enabled = False
        self._is_event_end = False
        self._done_condition = any
        self._race_duration = race_duration
        self._enable_domain_randomization = False

        # sqs client
        # The boto client errors out after polling for 1 hour.
        self._sqs_client = SQSClient(
            queue_url=self._queue_url,
            region_name=self._region,
            max_num_of_msg=MAX_NUM_OF_SQS_MESSAGE,
            wait_time_sec=SQS_WAIT_TIME_SEC,
            session=refreshed_session(self._region),
        )
        self._s3_client = S3Client(region_name=self._region)
        self._model_updater = ModelUpdater.get_instance()

        # tracking current state information
        self._track_data = TrackData.get_instance()
        self._start_lane = self._track_data.center_line
        # keep track of the racer specific info, e.g. s3 locations, alias, car color etc.
        self._current_racer = None
        # keep track of the current race car we are using, e.g. racecar_0
        self._current_car_model_state = None
        # keep track of the current control agent we are using
        self._current_agent = None
        # keep track of the current control graph manager
        self._current_graph_manager = None
        # Keep track of previous model's name
        self._prev_model_name = None
        self._park_position_idx = 0
        self._park_positions = get_hide_positions(len(SENSOR_MODEL_MAP))
        self._run_phase_subject = RunPhaseSubject()
        self._simtrace_video_s3_writers = []

        self._local_model_directory = "./checkpoint"

        # virtual event only have single agent, so set agent_name to "agent"
        self._agent_name = "agent"

        # camera manager
        self._camera_manager = CameraManager.get_instance()

        # setting up virtual event top and follow camera in CameraManager
        # virtual event configure camera does not need to wait for car to spawm because
        # follow car camera is not tracking any car initially
        self._main_cameras, self._sub_camera = configure_camera(
            namespaces=[VIRTUAL_EVENT], is_wait_for_model=False
        )

        # pop out all cameras after configuration to prevent camera from moving
        self._camera_manager.pop(namespace=VIRTUAL_EVENT)

        dummy_metrics_s3_config = {
            MetricsS3Keys.METRICS_BUCKET.value: "dummy-bucket",
            MetricsS3Keys.METRICS_KEY.value: "dummy-key",
            MetricsS3Keys.REGION.value: self._region,
        }

        self._eval_metrics = EvalMetrics(
            agent_name=self._agent_name,
            s3_dict_metrics=dummy_metrics_s3_config,
            is_continuous=self._is_continuous,
            pause_time_before_start=PAUSE_TIME_BEFORE_START,
        )

        # upload a default best sector time for all sectors with time inf for each sector
        # if there is not best sector time existed in s3

        # use the s3 bucket and prefix for yaml file stored as environment variable because
        # here is SimApp use only. For virtual event there is no s3 bucket and prefix past
        # through yaml file. All are past through sqs. For simplicity, reuse the yaml s3 bucket
        # and prefix environment variable.
        virtual_event_best_sector_time = VirtualEventBestSectorTime(
            bucket=os.environ.get("YAML_S3_BUCKET", ""),
            s3_key=get_s3_key(os.environ.get("YAML_S3_PREFIX", ""), SECTOR_TIME_S3_POSTFIX),
            region_name=os.environ.get("APP_REGION", "us-east-1"),
            local_path=SECTOR_TIME_LOCAL_PATH,
        )
        response = virtual_event_best_sector_time.list()
        # this is used to handle situation such as robomaker job crash, so the next robomaker job
        # can catch the best sector time left over from crashed job
        if "Contents" not in response:
            virtual_event_best_sector_time.persist(
                body=json.dumps(
                    {
                        SECTOR_X_FORMAT.format(idx + 1): float("inf")
                        for idx in range(self._num_sectors)
                    }
                ),
                s3_kms_extra_args=utils.get_s3_kms_extra_args(),
            )

        # ROS service to indicate all the robomaker markov packages are ready for consumption
        signal_robomaker_markov_package_ready()

        PhaseObserver("/agent/training_phase", self._run_phase_subject)

        # setup mp4 services
        self._setup_mp4_services()

    @property
    def current_racer(self):
        """Get the current racer object.

        Returns:
            RacerInformation: Information about current racer that was passed-in from the queue.
        """
        return self._current_racer

    @property
    def is_event_end(self):
        """Return True if the service has signaled event end

        Returns:
            boolean: Is it the time to kill everything and die
        """
        return self._is_event_end

    def poll_next_racer(self):
        """
        Poll from sqs the next racer information.
        """
        received_racer = False
        error_counter = 0
        while not received_racer:
            # Polling MAX_NUM_OF_SQS_MESSAGE=1 message from sqs
            # with wait time specified in SQS_WAIT_TIME_SEC
            response = self._sqs_client.get_messages()
            # If polling message is successful, it will return a list of payloads
            # If polling message failed, it will return an integer
            # 1=ClientError Or FailedToDeleteMessage
            # 2=SystemError
            if isinstance(response, int):
                error_counter += response
                if error_counter >= MAX_NUM_OF_SQS_ERROR:
                    # something went really wrong with the sqs queue...
                    log_and_exit(
                        "[virtual event manager] Too many exceptions (num={}) in \
                                 receiving message from sqs queue: {}".format(
                            error_counter, self._queue_url
                        ),
                        SIMAPP_SQS_RECEIVE_MESSAGE_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_500,
                    )
            elif isinstance(response, list) and len(response) == 1:
                message_body = response[0]
                try:
                    # validate the current racer information.
                    validate_json_input(message_body, RACER_INFO_JSON_SCHEMA)
                    # Parse JSON into an racer information object
                    # with attributes corresponding to dict keys
                    self._current_racer = json.loads(
                        message_body,
                        object_hook=lambda d: namedtuple(RACER_INFO_OBJECT, d.keys())(*d.values()),
                    )
                    # only set received_racer to True after making sure the message is valid.
                    received_racer = True
                    LOG.info(
                        "[virtual event manager] Received next racer's information %s",
                        self._current_racer,
                    )
                except GenericNonFatalException as ex:
                    ex.log_except_and_continue()

    def setup_race(self):
        """
            Setting up the race for the current racer.

        Returns:
            bool: True if setup race is successful.
                  False is a non fatal exception occurred.
        """

        LOG.info("[virtual event manager] Setting up race for racer")
        try:
            # unpause the physics in current world
            self._model_updater.unpause_physics()
            LOG.info("[virtual event manager] Unpaused physics in current world.")
            # set camera to starting position
            initial_pose = self._track_data.get_racecar_start_pose(
                racecar_idx=0, racer_num=1, start_position=get_start_positions(1)[0]
            )
            self._main_cameras[VIRTUAL_EVENT].reset_pose(car_pose=initial_pose)
            LOG.info("[virtual event manager] Reset camera to starting line.")
            if self._prev_model_name is not None:
                # NOTE: it's by design that we immediately part the previous car to pit
                # location right after unpause physics. This prevents any unwanted
                # leftover behavior to happen
                self._park_at_pit_location(self._prev_model_name)
                LOG.info(
                    "[virtual event manager] Parked previous model %s to pit location.",
                    self._prev_model_name,
                )
            self._model_updater.pause_physics()
            LOG.info("[virtual event manager] Paused physics in current world.")
            # download model metadata from s3
            sensors, version, model_metadata = self._download_model_metadata()
            # based on model metadata, select racecar
            self._current_car_model_state = self._get_car_model_state(sensors)
            # download checkpoint from s3
            checkpoint = self._download_checkpoint(version)
            # setup the simtrace and mp4 writers if the s3 locations are available
            self._setup_simtrace_mp4_writers()
            # reset the metrics s3 location for the current racer
            self._reset_metrics_loc()
            # setup agents
            agent_list = self._get_agent_list(model_metadata, version)
            self._setup_graph_manager(checkpoint, agent_list)
            LOG.info(
                "[virtual event manager] Graph manager successfully created the graph: setup race successful."
            )
            return True
        except GenericNonFatalException as ex:
            ex.log_except_and_continue()
            self.upload_race_status(
                status_code=ex.error_code, error_name=ex.error_name, error_details=ex.error_msg
            )
            self._clean_up_race()
            return False
        except Exception as ex:
            log_and_exit(
                "[virtual event manager] Something really wrong happened when setting up the race. {}".format(
                    ex
                ),
                SIMAPP_VIRTUAL_EVENT_RACE_EXCEPTION,
                SIMAPP_EVENT_ERROR_CODE_500,
            )

    def _download_checkpoint(self, version):
        """Setup the Checkpoint object and selete the best checkpoint.

        Args:
            version (float): SimApp version

        Returns:
            Checkpoint: Checkpoint class instance
        """
        # download checkpoint from s3
        checkpoint = Checkpoint(
            bucket=self._current_racer.inputModel.s3BucketName,
            s3_prefix=self._current_racer.inputModel.s3KeyPrefix,
            region_name=self._region,
            agent_name=self._agent_name,
            checkpoint_dir=self._local_model_directory,
        )
        # make coach checkpoint compatible
        if version < SIMAPP_VERSION_2 and not checkpoint.rl_coach_checkpoint.is_compatible():
            checkpoint.rl_coach_checkpoint.make_compatible(checkpoint.syncfile_ready)
        # get best model checkpoint string
        model_checkpoint_name = checkpoint.deepracer_checkpoint_json.get_deepracer_best_checkpoint()
        # Select the best checkpoint model by uploading rl coach .coach_checkpoint file
        model_kms = (
            self._current_racer.inputModel.s3KmsKeyArn
            if hasattr(self._current_racer.inputModel, "s3KmsKeyArn")
            else None
        )
        checkpoint.rl_coach_checkpoint.update(
            model_checkpoint_name=model_checkpoint_name,
            s3_kms_extra_args=utils.get_s3_extra_args(model_kms),
        )
        return checkpoint

    def _download_model_metadata(self):
        """Attempt to download model metadata from s3.

        Raises:
            GenericNonFatalException: An non fatal exception which we will
                                      catch and proceed with work loop.

        Returns:
            sensors, version, model_metadata: The needed information from model metadata.
        """
        model_metadata_s3_key = get_s3_key(
            self._current_racer.inputModel.s3KeyPrefix, MODEL_METADATA_S3_POSTFIX
        )
        try:
            model_metadata = ModelMetadata(
                bucket=self._current_racer.inputModel.s3BucketName,
                s3_key=model_metadata_s3_key,
                region_name=self._region,
                local_path=MODEL_METADATA_LOCAL_PATH_FORMAT.format(self._agent_name),
            )
            model_metadata_info = model_metadata.get_model_metadata_info()
            sensors = model_metadata_info[ModelMetadataKeys.SENSOR.value]
            simapp_version = model_metadata_info[ModelMetadataKeys.VERSION.value]
        except botocore.exceptions.ClientError as err:
            error_msg = "[s3] Client Error: Failed to download model_metadata file: \
                        s3_bucket: {}, s3_key: {}, {}.".format(
                self._current_racer.inputModel.s3BucketName, model_metadata_s3_key, err
            )
            raise GenericNonFatalException(
                error_msg=error_msg,
                error_code=SIMAPP_EVENT_ERROR_CODE_400,
                error_name=SIMAPP_EVENT_USER_ERROR,
            )
        except Exception as err:
            error_msg = "[s3] System Error: Failed to download model_metadata file: \
                        s3_bucket: {}, s3_key: {}, {}.".format(
                self._current_racer.inputModel.s3BucketName, model_metadata_s3_key, err
            )
            raise GenericNonFatalException(
                error_msg=error_msg,
                error_code=SIMAPP_EVENT_ERROR_CODE_500,
                error_name=SIMAPP_EVENT_SYSTEM_ERROR,
            )
        return sensors, simapp_version, model_metadata

    def start_race(self):
        """
        Start the race (evaluation) for the current racer.
        """
        LOG.info(
            "[virtual event manager] Starting race for racer %s", self._current_racer.racerAlias
        )
        # update the car on current model if does not use f1 or tron type of shell
        if const.F1 not in self._body_shell_type.lower():
            self._model_updater.update_model_color(
                self._current_car_model_state.model_name, self._current_racer.carConfig.carColor
            )
        # send request
        if self._is_save_mp4_enabled:
            self._subscribe_to_save_mp4(
                VirtualEventVideoEditSrvRequest(
                    display_name=self._current_racer.racerAlias,
                    racecar_color=self._current_racer.carConfig.carColor,
                )
            )

        # Update CameraManager by adding cameras into the current namespace. By doing so
        # a single follow car camera will follow the current active racecar.
        self._camera_manager.add(
            self._main_cameras[VIRTUAL_EVENT], self._current_car_model_state.model_name
        )
        self._camera_manager.add(self._sub_camera, self._current_car_model_state.model_name)

        configure_environment_randomizer()
        # strip index for park position
        self._park_position_idx = get_racecar_idx(self._current_car_model_state.model_name)
        # set the park position in track and do evaluation
        # Before each evaluation episode (single lap for non-continuous race and complete race for
        # continuous race), a new copy of park_positions needs to be loaded into track_data because
        # a park position will be pop from park_positions when a racer car need to be parked.
        # unpause the physics in current world
        self._model_updater.unpause_physics()
        LOG.info("[virtual event manager] Unpaused physics in current world.")
        if (
            self._prev_model_name is not None
            and self._prev_model_name != self._current_car_model_state.model_name
        ):
            # disable the links on the prev car
            # we are doing it here because we don't want the car to float around
            # after the link is disabled
            prev_car_model_state = ModelState()
            prev_car_model_state.model_name = self._prev_model_name
        LOG.info("[virtual event manager] Unpaused model for current car.")
        if self._is_continuous:
            self._track_data.park_positions = [self._park_positions[self._park_position_idx]]
            self._current_graph_manager.evaluate(EnvironmentSteps(1))
        else:
            for _ in range(self._number_of_trials):
                self._track_data.park_positions = [self._park_positions[self._park_position_idx]]
                self._current_graph_manager.evaluate(EnvironmentSteps(1))

    def finish_race(self):
        """
        Finish the race for the current racer.
        """
        # pause physics of the world
        self._model_updater.pause_physics()
        time.sleep(1)
        # unsubscribe mp4
        if self._is_save_mp4_enabled:
            self._unsubscribe_from_save_mp4(EmptyRequest())
        self._track_data.remove_object(name=self._current_car_model_state.model_name)
        LOG.info(
            "[virtual event manager] Finish Race - remove object %s.",
            self._current_car_model_state.model_name,
        )
        # pop out current racecar from camera namespace to prevent camera from moving
        self._camera_manager.pop(namespace=self._current_car_model_state.model_name)
        # upload simtrace and mp4 into s3 bucket
        self._save_simtrace_mp4()
        self.upload_race_status(status_code=200)
        # keep track of the previous model name
        self._prev_model_name = self._current_car_model_state.model_name
        # clean up local trace of current race
        self._clean_up_race()

    def _clean_up_race(self):
        """Helper function to clean up everything related to the ex-racer
        and get ready for the next racer.
        """
        self._simtrace_video_s3_writers = []
        # clean up local checkpoints etc.
        self._clean_local_directory()
        # reset the tensorflow graph to avoid errors with the global session
        tf.reset_default_graph()
        # reset the current racer
        self._current_racer = None
        self._current_agent = None
        self._current_car_model_state = None
        self._current_graph_manager = None
        self._park_position_idx = 0

    def _save_simtrace_mp4(self):
        """Get the appropriate kms key and save the simtrace and mp4 files."""
        # TODO: It might be theorically possible to have different kms keys for simtrace and mp4
        # but we are using the same key now since that's what happens in real life
        # consider refactor the simtrace_video_s3_writers later.
        if hasattr(self._current_racer.outputMp4, "s3KmsKeyArn"):
            simtrace_mp4_kms = self._current_racer.outputMp4.s3KmsKeyArn
        elif hasattr(self._current_racer.outputSimTrace, "s3KmsKeyArn"):
            simtrace_mp4_kms = self._current_racer.outputSimTrace.s3KmsKeyArn
        else:
            simtrace_mp4_kms = None
        for s3_writer in self._simtrace_video_s3_writers:
            s3_writer.persist(utils.get_s3_extra_args(simtrace_mp4_kms))

    def _reset_metrics_loc(self):
        """Reset the metrics location as new racer is loaded."""
        metrics_s3_config = {
            MetricsS3Keys.METRICS_BUCKET.value: self._current_racer.outputMetrics.s3BucketName,
            MetricsS3Keys.METRICS_KEY.value: self._current_racer.outputMetrics.s3KeyPrefix,
            MetricsS3Keys.REGION.value: self._region,
        }
        self._eval_metrics.reset_metrics(
            s3_dict_metrics=metrics_s3_config,
            is_save_simtrace_enabled=self._is_save_simtrace_enabled,
        )

    def _setup_simtrace_mp4_writers(self):
        """Setup the simtrace and mp4 writers if the locations are passed in."""
        self._is_save_simtrace_enabled = False
        self._is_save_mp4_enabled = False
        if hasattr(self._current_racer, "outputSimTrace"):
            self._simtrace_video_s3_writers.append(
                SimtraceVideo(
                    upload_type=SimtraceVideoNames.SIMTRACE_EVAL.value,
                    bucket=self._current_racer.outputSimTrace.s3BucketName,
                    s3_prefix=self._current_racer.outputSimTrace.s3KeyPrefix,
                    region_name=self._region,
                    local_path=SIMTRACE_EVAL_LOCAL_PATH_FORMAT.format(self._agent_name),
                )
            )
            self._is_save_simtrace_enabled = True
        if hasattr(self._current_racer, "outputMp4"):
            self._simtrace_video_s3_writers.extend(
                [
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.PIP.value,
                        bucket=self._current_racer.outputMp4.s3BucketName,
                        s3_prefix=self._current_racer.outputMp4.s3KeyPrefix,
                        region_name=self._region,
                        local_path=CAMERA_PIP_MP4_LOCAL_PATH_FORMAT.format(self._agent_name),
                    ),
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.DEGREE45.value,
                        bucket=self._current_racer.outputMp4.s3BucketName,
                        s3_prefix=self._current_racer.outputMp4.s3KeyPrefix,
                        region_name=self._region,
                        local_path=CAMERA_45DEGREE_LOCAL_PATH_FORMAT.format(self._agent_name),
                    ),
                    SimtraceVideo(
                        upload_type=SimtraceVideoNames.TOPVIEW.value,
                        bucket=self._current_racer.outputMp4.s3BucketName,
                        s3_prefix=self._current_racer.outputMp4.s3KeyPrefix,
                        region_name=self._region,
                        local_path=CAMERA_TOPVIEW_LOCAL_PATH_FORMAT.format(self._agent_name),
                    ),
                ]
            )
            self._is_save_mp4_enabled = True

    def _setup_graph_manager(self, checkpoint, agent_list):
        """Sets up graph manager based on the checkpoint file and agents list.

        Args:
            checkpoint (Checkpoint): The model checkpoints we just downloaded.
            agent_list (list[Agent]): List of agents we want to setup graph manager for.
        """
        sm_hyperparams_dict = {}
        self._current_graph_manager, _ = get_graph_manager(
            hp_dict=sm_hyperparams_dict,
            agent_list=agent_list,
            run_phase_subject=self._run_phase_subject,
            enable_domain_randomization=self._enable_domain_randomization,
            done_condition=self._done_condition,
        )
        checkpoint_dict = dict()
        checkpoint_dict[self._agent_name] = checkpoint
        ds_params_instance = S3BotoDataStoreParameters(checkpoint_dict=checkpoint_dict)

        self._current_graph_manager.data_store = S3BotoDataStore(
            params=ds_params_instance,
            graph_manager=self._current_graph_manager,
            ignore_lock=True,
            log_and_cont=True,
        )
        self._current_graph_manager.env_params.seed = 0

        self._current_graph_manager.data_store.wait_for_checkpoints()
        self._current_graph_manager.data_store.modify_checkpoint_variables()

        task_parameters = TaskParameters()
        task_parameters.checkpoint_restore_path = self._local_model_directory

        self._current_graph_manager.create_graph(
            task_parameters=task_parameters,
            stop_physics=self._model_updater.pause_physics_service,
            start_physics=self._model_updater.unpause_physics_service,
            empty_service_call=EmptyRequest,
        )

    def _get_agent_list(self, model_metadata, version):
        """Setup agent and get the agents list.

        Args:
            model_metadata (ModelMetadata): Current racer's model metadata
            version (str): The current racer's simapp version in the model metadata

        Returns:
            agent_list (list): The list of agents for the current racer
        """
        # setup agent
        agent_config = {
            "model_metadata": model_metadata,
            ConfigParams.CAR_CTRL_CONFIG.value: {
                ConfigParams.LINK_NAME_LIST.value: [
                    link_name.replace("racecar", self._current_car_model_state.model_name)
                    for link_name in LINK_NAMES
                ],
                ConfigParams.VELOCITY_LIST.value: [
                    velocity_topic.replace("racecar", self._current_car_model_state.model_name)
                    for velocity_topic in VELOCITY_TOPICS
                ],
                ConfigParams.STEERING_LIST.value: [
                    steering_topic.replace("racecar", self._current_car_model_state.model_name)
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
                ConfigParams.AGENT_NAME.value: self._current_car_model_state.model_name,
                ConfigParams.VERSION.value: version,
                ConfigParams.NUMBER_OF_RESETS.value: self._number_of_resets,
                ConfigParams.PENALTY_SECONDS.value: self._penalty_seconds,
                ConfigParams.NUMBER_OF_TRIALS.value: self._number_of_trials,
                ConfigParams.IS_CONTINUOUS.value: self._is_continuous,
                ConfigParams.RACE_TYPE.value: self._race_type,
                ConfigParams.COLLISION_PENALTY.value: self._collision_penalty,
                ConfigParams.OFF_TRACK_PENALTY.value: self._off_track_penalty,
                ConfigParams.START_POSITION.value: get_start_positions(1)[
                    0
                ],  # hard-coded to the first start position
                ConfigParams.DONE_CONDITION.value: self._done_condition,
                ConfigParams.IS_VIRTUAL_EVENT.value: True,
                ConfigParams.RACE_DURATION.value: self._race_duration,
            },
        }

        agent_list = list()
        agent_list.append(
            create_rollout_agent(agent_config, self._eval_metrics, self._run_phase_subject)
        )
        agent_list.append(create_obstacles_agent())
        agent_list.append(create_bot_cars_agent())
        return agent_list

    def _setup_mp4_services(self):
        """
        Setting up the mp4 ros services if mp4s need to be saved.
        """
        mp4_sub = "/{}/save_mp4/subscribe_to_save_mp4".format(VIRTUAL_EVENT)
        mp4_unsub = "/{}/save_mp4/unsubscribe_from_save_mp4".format(VIRTUAL_EVENT)
        rospy.wait_for_service(mp4_sub)
        rospy.wait_for_service(mp4_unsub)
        self._subscribe_to_save_mp4 = ServiceProxyWrapper(mp4_sub, VirtualEventVideoEditSrv)
        self._unsubscribe_from_save_mp4 = ServiceProxyWrapper(mp4_unsub, Empty)

    def _get_car_model_state(self, sensors: list) -> ModelState:
        """Get the current car model state according to sensors configuration.

        Args:
            sensors (list): sensors in the model metadata

        Returns:
            ModelState: a model state object with the current racecar name
        """
        is_stereo = False
        is_lidar = False
        if Input.STEREO.value in sensors:
            is_stereo = True
            LOG.info("[virtual event manager] stereo camera present")
        if Input.LIDAR.value in sensors or Input.SECTOR_LIDAR.value in sensors:
            is_lidar = True
            LOG.info("[virtual event manager] lidar present")
        car_model_state = ModelState()
        if is_stereo:
            if is_lidar:
                car_model_state.model_name = SENSOR_MODEL_MAP["stereo_camera_lidar"]
            else:
                car_model_state.model_name = SENSOR_MODEL_MAP["stereo_camera"]
        else:
            if is_lidar:
                car_model_state.model_name = SENSOR_MODEL_MAP["single_camera_lidar"]
            else:
                car_model_state.model_name = SENSOR_MODEL_MAP["single_camera"]
        return car_model_state

    def _clean_local_directory(self):
        """Clean up the local directory after race ends."""
        LOG.info("[virtual event manager] cleaning up the local directory after race ends.")
        for root, _, files in os.walk(self._local_model_directory):
            for f in files:
                os.remove(os.path.join(root, f))

    def _park_at_pit_location(self, model_name):
        """Reset car to inital position."""
        # set the car at the pit parking position
        yaw = 0.0 if self._track_data.is_ccw else math.pi
        self._model_updater.set_model_position(
            model_name, self._park_positions[self._park_position_idx], yaw, is_blocking=True
        )
        LOG.info("[virtual event manager] parked car to pit position.")

    def upload_race_status(self, status_code, error_name=None, error_details=None):
        """Upload race status into s3.

        Args:
            status_code (str): Status code for race.
            error_name (str, optional): The name of the error if is 4xx or 5xx.
                                        Defaults to "".
            error_details (str, optional): The detail message of the error
                                           if is 4xx or 5xx.
                                           Defaults to "".
        """
        # persist s3 status file
        if error_name is not None and error_details is not None:
            status = {
                RaceStatusKeys.STATUS_CODE.value: status_code,
                RaceStatusKeys.ERROR_NAME.value: error_name,
                RaceStatusKeys.ERROR_DETAILS.value: error_details,
            }
        else:
            status = {RaceStatusKeys.STATUS_CODE.value: status_code}
        status_json = json.dumps(status)
        s3_key = os.path.normpath(
            os.path.join(self._current_racer.outputStatus.s3KeyPrefix, S3_RACE_STATUS_FILE_NAME)
        )
        race_status_kms = (
            self._current_racer.outputStatus.s3KmsKeyArn
            if hasattr(self._current_racer.outputStatus, "s3KmsKeyArn")
            else None
        )
        self._s3_client.upload_fileobj(
            bucket=self._current_racer.outputStatus.s3BucketName,
            s3_key=s3_key,
            fileobj=io.BytesIO(status_json.encode()),
            s3_kms_extra_args=utils.get_s3_extra_args(race_status_kms),
        )
        LOG.info(
            "[virtual event manager] Successfully uploaded race status file to \
                 s3 bucket {} with s3 key {}.".format(
                self._current_racer.outputStatus.s3BucketName, s3_key
            )
        )
