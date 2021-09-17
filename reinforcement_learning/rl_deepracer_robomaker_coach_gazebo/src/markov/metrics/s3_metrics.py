"""This module houses the metric objects for the sim app"""
import json
import logging
import math
import os
import statistics
import time
from collections import OrderedDict

import boto3
import botocore
import rospy
from deepracer_simulation_environment.srv import VideoMetricsSrv, VideoMetricsSrvResponse
from geometry_msgs.msg import Point32
from markov.boto.s3.constants import (
    SIMTRACE_EVAL_LOCAL_PATH_FORMAT,
    SIMTRACE_TRAINING_LOCAL_PATH_FORMAT,
)
from markov.boto.s3.files.metrics import Metrics
from markov.common import ObserverInterface
from markov.constants import BEST_CHECKPOINT, LAST_CHECKPOINT, METRICS_VERSION
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.gazebo_tracker.constants import TrackerPriority
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger
from markov.metrics.constants import (
    BestModelMetricType,
    EpisodeStatus,
    MetricsS3Keys,
    Mp4VideoMetrics,
    StepMetrics,
)
from markov.metrics.metrics_interface import MetricsInterface
from markov.track_geom.track_data import TrackData
from markov.utils import get_boto_config, get_s3_kms_extra_args
from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.core_types import RunPhase

LOGGER = Logger(__name__, logging.INFO).get_logger()


#! TODO this needs to be removed after muti part is fixed, note we don't have
# agent name here, but we can add it to the step metrics if needed
def sim_trace_log(sim_trace_dict):
    """Logs the step metrics to cloud watch
    sim_trace_dict - Ordered dict containing the step metrics, note order must match
                     precision in the string
    """
    LOGGER.info(
        "SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%s,%.4f,%s,%s,%.4f,%d,%.2f,%s,%s,%.2f\n"
        % (tuple(sim_trace_dict.values()))
    )


def write_simtrace_to_local_file(file_path: str, metrics_data: OrderedDict):
    """Write the metrics data to s3
    Arguments:
        file_path {str} -- [description]
        metrics_data {OrderedDict} -- [description]
    """
    assert isinstance(
        metrics_data, OrderedDict
    ), "SimTrace metrics data argument must be of type OrderedDict"
    if metrics_data is not None:
        if not os.path.exists(file_path):
            with open(file_path, "w") as filepointer:
                filepointer.write(
                    ",".join([str(key) for key, value in metrics_data.items()]) + "\n"
                )
        with open(file_path, "a") as filepointer:
            filepointer.write(",".join([str(value) for key, value in metrics_data.items()]) + "\n")


class TrainingMetrics(MetricsInterface, ObserverInterface, AbstractTracker):
    """This class is responsible for uploading training metrics to s3"""

    def __init__(
        self,
        agent_name,
        s3_dict_metrics,
        deepracer_checkpoint_json,
        ckpnt_dir,
        run_phase_sink,
        use_model_picker=True,
    ):
        """s3_dict_metrics - Dictionary containing the required s3 info for the metrics
                          bucket with keys specified by MetricsS3Keys
        deepracer_checkpoint_json - DeepracerCheckpointJson instance
        ckpnt_dir - Directory where the current checkpont is to be stored
        run_phase_sink - Sink to recieve notification of a change in run phase
        use_model_picker - Flag to whether to use model picker or not.
        """
        self._agent_name_ = agent_name
        self._deepracer_checkpoint_json = deepracer_checkpoint_json
        self._s3_metrics = Metrics(
            bucket=s3_dict_metrics[MetricsS3Keys.METRICS_BUCKET.value],
            s3_key=s3_dict_metrics[MetricsS3Keys.METRICS_KEY.value],
            region_name=s3_dict_metrics[MetricsS3Keys.REGION.value],
        )
        self._start_time_ = time.time()
        self._episode_ = 0
        self._episode_reward_ = 0.0
        self._progress_ = 0.0
        self._episode_status = ""
        self._metrics_ = list()
        self._is_eval_ = True
        self._eval_trials_ = 0
        self._checkpoint_state_ = CheckpointStateFile(ckpnt_dir)
        self._use_model_picker = use_model_picker
        self._eval_stats_dict_ = {"chkpnt_name": None, "avg_eval_metric": None}
        self._best_chkpnt_stats = {"name": None, "avg_eval_metric": None, "time_stamp": time.time()}
        self._current_eval_best_model_metric_list_ = list()
        self.is_save_simtrace_enabled = rospy.get_param("SIMTRACE_S3_BUCKET", None)
        self._best_model_metric_type = BestModelMetricType(
            rospy.get_param("BEST_MODEL_METRIC", BestModelMetricType.PROGRESS.value).lower()
        )
        self.track_data = TrackData.get_instance()
        run_phase_sink.register(self)
        # Create the agent specific directories needed for storing the metric files
        self._simtrace_local_path = SIMTRACE_TRAINING_LOCAL_PATH_FORMAT.format(self._agent_name_)
        simtrace_dirname = os.path.dirname(self._simtrace_local_path)
        if simtrace_dirname or not os.path.exists(simtrace_dirname):
            os.makedirs(simtrace_dirname)
        self._current_sim_time = 0
        rospy.Service(
            "/{}/{}".format(self._agent_name_, "mp4_video_metrics"),
            VideoMetricsSrv,
            self._handle_get_video_metrics,
        )
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        AbstractTracker.__init__(self, TrackerPriority.HIGH)

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        self._current_sim_time = sim_time.clock.secs + 1.0e-9 * sim_time.clock.nsecs

    def reset(self):
        self._start_time_ = self._current_sim_time
        self._episode_reward_ = 0.0
        self._progress_ = 0.0

    def append_episode_metrics(self):
        self._episode_ += 1 if not self._is_eval_ else 0
        self._eval_trials_ += 1 if not self._is_eval_ else 0
        training_metric = dict()
        training_metric["reward_score"] = int(round(self._episode_reward_))
        training_metric["metric_time"] = int(round(self._current_sim_time * 1000))
        training_metric["start_time"] = int(round(self._start_time_ * 1000))
        training_metric["elapsed_time_in_milliseconds"] = int(
            round((self._current_sim_time - self._start_time_) * 1000)
        )
        training_metric["episode"] = int(self._episode_)
        training_metric["trial"] = int(self._eval_trials_)
        training_metric["phase"] = "evaluation" if self._is_eval_ else "training"
        training_metric["completion_percentage"] = int(self._progress_)
        training_metric["episode_status"] = EpisodeStatus.get_episode_status_label(
            self._episode_status
        )
        self._metrics_.append(training_metric)

    def upload_episode_metrics(self):
        json_metrics = json.dumps(
            {
                "metrics": self._metrics_,
                "version": METRICS_VERSION,
                "best_model_metric": self._best_model_metric_type.value,
            }
        )
        self._s3_metrics.persist(body=json_metrics, s3_kms_extra_args=get_s3_kms_extra_args())
        if self._is_eval_:
            if self._best_model_metric_type == BestModelMetricType.REWARD:
                self._current_eval_best_model_metric_list_.append(self._episode_reward_)
            else:
                self._current_eval_best_model_metric_list_.append(self._progress_)

    def upload_step_metrics(self, metrics):
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        self._episode_reward_ += metrics[StepMetrics.REWARD.value]
        if not self._is_eval_:
            metrics[StepMetrics.EPISODE.value] = self._episode_
            StepMetrics.validate_dict(metrics)
            sim_trace_log(metrics)
            if self.is_save_simtrace_enabled:
                write_simtrace_to_local_file(self._simtrace_local_path, metrics)

    def update(self, data):
        self._is_eval_ = data != RunPhase.TRAIN

        if not self._is_eval_ and self._use_model_picker:
            if self._eval_stats_dict_["chkpnt_name"] is None:
                self._eval_stats_dict_["chkpnt_name"] = self._checkpoint_state_.read().name

            self._eval_trials_ = 0
            mean_metric = (
                statistics.mean(self._current_eval_best_model_metric_list_)
                if self._current_eval_best_model_metric_list_
                else None
            )
            msg_format = "[BestModelSelection] Number of evaluations: {} Evaluation episode {}: {}"
            LOGGER.info(
                msg_format.format(
                    len(self._current_eval_best_model_metric_list_),
                    self._best_model_metric_type.value,
                    self._current_eval_best_model_metric_list_,
                )
            )
            LOGGER.info(
                "[BestModelSelection] Evaluation episode {} mean: {}".format(
                    self._best_model_metric_type.value, mean_metric
                )
            )
            self._current_eval_best_model_metric_list_.clear()

            time_stamp = self._current_sim_time
            if (
                self._eval_stats_dict_["avg_eval_metric"] is None
                or mean_metric >= self._eval_stats_dict_["avg_eval_metric"]
            ):
                msg_format = "[BestModelSelection] current {0} mean: {1} >= best {0} mean: {2}"
                LOGGER.info(
                    msg_format.format(
                        self._best_model_metric_type.value,
                        mean_metric,
                        self._eval_stats_dict_["avg_eval_metric"],
                    )
                )
                msg_format = '[BestModelSelection] Updating the best checkpoint to "{}" from "{}".'
                LOGGER.info(
                    msg_format.format(
                        self._eval_stats_dict_["chkpnt_name"], self._best_chkpnt_stats["name"]
                    )
                )
                self._eval_stats_dict_["avg_eval_metric"] = mean_metric
                self._best_chkpnt_stats = {
                    "name": self._eval_stats_dict_["chkpnt_name"],
                    "avg_eval_metric": mean_metric,
                    "time_stamp": time_stamp,
                }
            last_chkpnt_stats = {
                "name": self._eval_stats_dict_["chkpnt_name"],
                "avg_eval_metric": mean_metric,
                "time_stamp": time_stamp,
            }
            self._deepracer_checkpoint_json.persist(
                body=json.dumps(
                    {BEST_CHECKPOINT: self._best_chkpnt_stats, LAST_CHECKPOINT: last_chkpnt_stats}
                ),
                s3_kms_extra_args=get_s3_kms_extra_args(),
            )
            # Update the checkpoint name to the new checkpoint being used for training that will
            # then be evaluated, note this class gets notfied when the system is put into a
            # training phase and assumes that a training phase only starts when a new check point
            # is avaialble
            self._eval_stats_dict_["chkpnt_name"] = self._checkpoint_state_.read().name

    def update_mp4_video_metrics(self, metrics):
        agent_x, agent_y = metrics[StepMetrics.X.value], metrics[StepMetrics.Y.value]
        self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value] = 0
        self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value] = self._progress_
        # For continuous race, MP4 video will display the total reset counter for the entire race
        # For non-continuous race, MP4 video will display reset counter per lap
        self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value] = 0

        self._video_metrics[Mp4VideoMetrics.THROTTLE.value] = 0
        self._video_metrics[Mp4VideoMetrics.STEERING.value] = 0
        self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value] = 0
        self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value] = 0
        self._video_metrics[Mp4VideoMetrics.DONE.value] = metrics[StepMetrics.DONE.value]
        self._video_metrics[Mp4VideoMetrics.X.value] = agent_x
        self._video_metrics[Mp4VideoMetrics.Y.value] = agent_y

        object_poses = [
            pose
            for object_name, pose in self.track_data.object_poses.items()
            if not object_name.startswith("racecar")
        ]
        object_locations = []
        for pose in object_poses:
            point = Point32()
            point.x, point.y, point.z = pose.position.x, pose.position.y, 0
            object_locations.append(point)
        self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value] = object_locations

    def _handle_get_video_metrics(self, req):
        return VideoMetricsSrvResponse(
            self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value],
            self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value],
            self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value],
            self._video_metrics[Mp4VideoMetrics.THROTTLE.value],
            self._video_metrics[Mp4VideoMetrics.STEERING.value],
            self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value],
            self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value],
            self._video_metrics[Mp4VideoMetrics.DONE.value],
            self._video_metrics[Mp4VideoMetrics.X.value],
            self._video_metrics[Mp4VideoMetrics.Y.value],
            self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value],
            self._video_metrics[Mp4VideoMetrics.EPISODE_STATUS.value],
            self._video_metrics[Mp4VideoMetrics.PAUSE_DURATION.value],
        )


class EvalMetrics(MetricsInterface, AbstractTracker):
    """This class is responsible for uploading eval metrics to s3"""

    def __init__(self, agent_name, s3_dict_metrics, is_continuous, pause_time_before_start=0.0):
        """Init eval metrics

        Args:
            agent_name (string): agent name
            s3_dict_metrics (dict): Dictionary containing the required
                s3 info for the metrics bucket with keys specified by MetricsS3Keys
            is_continuous (bool): True if continuous race, False otherwise
            pause_time_before_start (float): second to pause before race start
        """
        self._pause_time_before_start = pause_time_before_start
        self._is_pause_time_subtracted = False
        self._agent_name_ = agent_name
        self._s3_metrics = Metrics(
            bucket=s3_dict_metrics[MetricsS3Keys.METRICS_BUCKET.value],
            s3_key=s3_dict_metrics[MetricsS3Keys.METRICS_KEY.value],
            region_name=s3_dict_metrics[MetricsS3Keys.REGION.value],
        )
        self._is_continuous = is_continuous
        self._start_time_ = time.time()
        self._number_of_trials_ = 0
        self._progress_ = 0.0
        self._episode_status = ""
        self._metrics_ = list()
        # This is used to calculate the actual distance traveled by the car
        self._agent_xy = list()
        self._prev_step_time = time.time()
        self.is_save_simtrace_enabled = rospy.get_param("SIMTRACE_S3_BUCKET", None)
        # Create the agent specific directories needed for storing the metric files
        self._simtrace_local_path = SIMTRACE_EVAL_LOCAL_PATH_FORMAT.format(self._agent_name_)
        simtrace_dirname = os.path.dirname(self._simtrace_local_path)
        if simtrace_dirname or not os.path.exists(simtrace_dirname):
            os.makedirs(simtrace_dirname)
        self.reset_count_dict = {
            EpisodeStatus.CRASHED.value: 0,
            EpisodeStatus.OFF_TRACK.value: 0,
            EpisodeStatus.IMMOBILIZED.value: 0,
            EpisodeStatus.REVERSED.value: 0,
        }
        self._best_lap_time = float("inf")
        self._total_evaluation_time = 0
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        self._reset_count_sum = 0
        self._current_sim_time = 0
        self.track_data = TrackData.get_instance()
        rospy.Service(
            "/{}/{}".format(self._agent_name_, "mp4_video_metrics"),
            VideoMetricsSrv,
            self._handle_get_video_metrics,
        )
        AbstractTracker.__init__(self, TrackerPriority.HIGH)

    def reset_metrics(self, s3_dict_metrics, is_save_simtrace_enabled):
        """reset matrics for virtual event when next racer coming in

        Args:
            s3_dict_metrics (dict): dictionary for s3 metrics
            is_save_simtrace_enabled (bool): True if saving simtrace. False, otherwise.
        """
        self._s3_metrics = Metrics(
            bucket=s3_dict_metrics[MetricsS3Keys.METRICS_BUCKET.value],
            s3_key=s3_dict_metrics[MetricsS3Keys.METRICS_KEY.value],
            region_name=s3_dict_metrics[MetricsS3Keys.REGION.value],
        )
        self.is_save_simtrace_enabled = is_save_simtrace_enabled
        self.clear()

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        self._current_sim_time = sim_time.clock.secs + 1.0e-9 * sim_time.clock.nsecs

    def reset(self):
        """clear reset counter only for non-continuous racing"""
        # for continuous race: add pause time for only first lap
        if self._is_continuous:
            self._start_time_ = self._current_sim_time
            if not self._is_pause_time_subtracted:
                self._start_time_ += self._pause_time_before_start
                self._is_pause_time_subtracted = True
        # for non-continuous race: add pause time for every lap
        else:
            self._start_time_ = self._current_sim_time + self._pause_time_before_start
        self._reset_count_sum += (
            self.reset_count_dict[EpisodeStatus.CRASHED.value]
            + self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value]
            + self.reset_count_dict[EpisodeStatus.OFF_TRACK.value]
            + self.reset_count_dict[EpisodeStatus.REVERSED.value]
        )
        for key in self.reset_count_dict.keys():
            self.reset_count_dict[key] = 0

    def append_episode_metrics(self, is_complete=True):
        if not is_complete and self._number_of_trials_ != 0:
            # Note: for virtual event, if the racer did not even finish one lap
            # for the duration of the event, we display DNF.
            # However, our friends at the game want the DNF ranks as well
            # so we append the incomplete metrics for ppl who didn't finish
            # first lap
            LOGGER.info(
                "Appending episode metrics for incomplete lap skipped, laps completed %s",
                self._number_of_trials_,
            )
            return
        eval_metric = dict()
        eval_metric["completion_percentage"] = int(self._progress_)
        eval_metric["metric_time"] = int(round(self._current_sim_time * 1000))
        eval_metric["start_time"] = int(round(self._start_time_ * 1000))
        eval_metric["elapsed_time_in_milliseconds"] = int(
            round((self._current_sim_time - self._start_time_) * 1000)
        )
        eval_metric["episode_status"] = EpisodeStatus.get_episode_status_label(self._episode_status)
        eval_metric["crash_count"] = self.reset_count_dict[EpisodeStatus.CRASHED.value]
        eval_metric["immobilized_count"] = self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value]
        eval_metric["off_track_count"] = self.reset_count_dict[EpisodeStatus.OFF_TRACK.value]
        eval_metric["reversed_count"] = self.reset_count_dict[EpisodeStatus.REVERSED.value]
        eval_metric["reset_count"] = (
            eval_metric["crash_count"]
            + eval_metric["immobilized_count"]
            + eval_metric["off_track_count"]
            + eval_metric["reversed_count"]
        )
        if is_complete:
            self._number_of_trials_ += 1
            self._best_lap_time = min(
                eval_metric["elapsed_time_in_milliseconds"], self._best_lap_time
            )
            self._total_evaluation_time += eval_metric["elapsed_time_in_milliseconds"]
        eval_metric["trial"] = int(self._number_of_trials_)
        self._metrics_.append(eval_metric)

    def upload_episode_metrics(self):
        # TODO: Service team can't handle "version" key in Evaluation Metrics due to
        # unknown key in the json. Training metrics change works fine as the metrics.json
        # file is directly loaded in Front-end Console while evaluation metrics file is loaded through
        # Service API and Service can't handle the keys in metrics file that is not defined in the service.
        # Keeping evaluation metrics as it is (without version key) as there is no change in the format anyway.
        # But we should make change in the future to match the format with Training metrics.
        json_metrics = json.dumps({"metrics": self._metrics_})
        self._s3_metrics.persist(body=json_metrics, s3_kms_extra_args=get_s3_kms_extra_args())

    def update_mp4_video_metrics(self, metrics):
        actual_speed = 0
        cur_time = self._current_sim_time
        agent_x, agent_y = metrics[StepMetrics.X.value], metrics[StepMetrics.Y.value]
        if self._agent_xy:
            # Speed = Distance/Time
            delta_time = cur_time - self._prev_step_time
            actual_speed = 0
            if delta_time:
                actual_speed = (
                    math.sqrt(
                        (self._agent_xy[0] - agent_x) ** 2 + (self._agent_xy[1] - agent_y) ** 2
                    )
                    / delta_time
                )
        self._agent_xy = [agent_x, agent_y]
        self._prev_step_time = cur_time

        self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value] = self._number_of_trials_
        self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value] = self._progress_
        # For continuous race, MP4 video will display the total reset counter for the entire race
        # For non-continuous race, MP4 video will display reset counter per lap
        self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value] = (
            self.reset_count_dict[EpisodeStatus.CRASHED.value]
            + self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value]
            + self.reset_count_dict[EpisodeStatus.OFF_TRACK.value]
            + self.reset_count_dict[EpisodeStatus.REVERSED.value]
            + (self._reset_count_sum if self._is_continuous else 0)
        )

        self._video_metrics[Mp4VideoMetrics.THROTTLE.value] = actual_speed
        self._video_metrics[Mp4VideoMetrics.STEERING.value] = metrics[StepMetrics.STEER.value]
        self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value] = self._best_lap_time
        self._video_metrics[
            Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value
        ] = self._total_evaluation_time + int(
            round((self._current_sim_time - self._start_time_) * 1000)
        )
        self._video_metrics[Mp4VideoMetrics.DONE.value] = metrics[StepMetrics.DONE.value]
        self._video_metrics[Mp4VideoMetrics.X.value] = agent_x
        self._video_metrics[Mp4VideoMetrics.Y.value] = agent_y

        object_poses = [
            pose
            for object_name, pose in self.track_data.object_poses.items()
            if not object_name.startswith("racecar")
        ]
        object_locations = []
        for pose in object_poses:
            point = Point32()
            point.x, point.y, point.z = pose.position.x, pose.position.y, 0
            object_locations.append(point)
        self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value] = object_locations
        self._video_metrics[Mp4VideoMetrics.EPISODE_STATUS.value] = metrics[
            StepMetrics.EPISODE_STATUS.value
        ]
        self._video_metrics[Mp4VideoMetrics.PAUSE_DURATION.value] = metrics[
            StepMetrics.PAUSE_DURATION.value
        ]

    def upload_step_metrics(self, metrics):
        metrics[StepMetrics.EPISODE.value] = self._number_of_trials_
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        if self._episode_status in self.reset_count_dict:
            self.reset_count_dict[self._episode_status] += 1
        StepMetrics.validate_dict(metrics)
        sim_trace_log(metrics)
        if self.is_save_simtrace_enabled:
            write_simtrace_to_local_file(self._simtrace_local_path, metrics)

    def _handle_get_video_metrics(self, req):
        return VideoMetricsSrvResponse(
            self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value],
            self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value],
            self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value],
            self._video_metrics[Mp4VideoMetrics.THROTTLE.value],
            self._video_metrics[Mp4VideoMetrics.STEERING.value],
            self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value],
            self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value],
            self._video_metrics[Mp4VideoMetrics.DONE.value],
            self._video_metrics[Mp4VideoMetrics.X.value],
            self._video_metrics[Mp4VideoMetrics.Y.value],
            self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value],
            self._video_metrics[Mp4VideoMetrics.EPISODE_STATUS.value],
            self._video_metrics[Mp4VideoMetrics.PAUSE_DURATION.value],
        )

    def clear(self):
        """clear all EvalMetrics member variable"""
        self._is_pause_time_subtracted = False
        self._start_time_ = self._current_sim_time
        self._number_of_trials_ = 0
        self._progress_ = 0.0
        self._episode_status = ""
        self._metrics_ = list()
        self._agent_xy = list()
        self._prev_step_time = self._current_sim_time
        self.reset_count_dict = {
            EpisodeStatus.CRASHED.value: 0,
            EpisodeStatus.OFF_TRACK.value: 0,
            EpisodeStatus.IMMOBILIZED.value: 0,
            EpisodeStatus.REVERSED.value: 0,
        }
        self._best_lap_time = float("inf")
        self._total_evaluation_time = 0
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        self._reset_count_sum = 0
        self._current_sim_time = 0
