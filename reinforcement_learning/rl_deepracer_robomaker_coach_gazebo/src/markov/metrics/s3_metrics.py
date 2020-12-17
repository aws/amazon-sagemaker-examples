'''This module houses the metric objects for the sim app'''
import math
import time
import json
import logging
import os
from collections import OrderedDict
import statistics
import boto3
import botocore
import rospy
from deepracer_simulation_environment.srv import VideoMetricsSrvResponse, VideoMetricsSrv
from geometry_msgs.msg import Point32
from markov.constants import BEST_CHECKPOINT, LAST_CHECKPOINT
from markov.common import ObserverInterface
from markov.metrics.constants import (MetricsS3Keys, StepMetrics, EpisodeStatus,
                                      Mp4VideoMetrics)
from markov.metrics.metrics_interface import MetricsInterface
from markov.utils import get_boto_config, get_s3_kms_extra_args
from markov.log_handler.logger import Logger
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.constants import (SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500)
from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.core_types import RunPhase
from markov.gazebo_tracker.abs_tracker import AbstractTracker
from markov.gazebo_tracker.constants import TrackerPriority
from markov.track_geom.track_data import TrackData
from markov.s3.constants import (SIMTRACE_EVAL_LOCAL_PATH_FORMAT,
                                 SIMTRACE_TRAINING_LOCAL_PATH_FORMAT)
from markov.s3.files.metrics import Metrics

LOGGER = Logger(__name__, logging.INFO).get_logger()

#! TODO this needs to be removed after muti part is fixed, note we don't have
# agent name here, but we can add it to the step metrics if needed
def sim_trace_log(sim_trace_dict):
    '''Logs the step metrics to cloud watch
       sim_trace_dict - Ordered dict containing the step metrics, note order must match
                        precision in the string
    '''
    LOGGER.info('SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s,%s\n' % \
        (tuple(sim_trace_dict.values())))

def write_simtrace_to_local_file(file_path: str, metrics_data: OrderedDict):
    """ Write the metrics data to s3
    Arguments:
        file_path {str} -- [description]
        metrics_data {OrderedDict} -- [description]
    """
    assert isinstance(metrics_data, OrderedDict), 'SimTrace metrics data argument must be of type OrderedDict'
    if metrics_data is not None:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as filepointer:
                filepointer.write(','.join([str(key) for key, value in metrics_data.items()])+"\n")
        with open(file_path, 'a') as filepointer:
            filepointer.write(','.join([str(value) for key, value in metrics_data.items()])+"\n")

class TrainingMetrics(MetricsInterface, ObserverInterface, AbstractTracker):
    '''This class is responsible for uploading training metrics to s3'''
    def __init__(self, agent_name, s3_dict_metrics, deepracer_checkpoint_json, ckpnt_dir, run_phase_sink, use_model_picker=True):
        '''s3_dict_metrics - Dictionary containing the required s3 info for the metrics
                             bucket with keys specified by MetricsS3Keys
           deepracer_checkpoint_json - DeepracerCheckpointJson instance
           ckpnt_dir - Directory where the current checkpont is to be stored
           run_phase_sink - Sink to recieve notification of a change in run phase
           use_model_picker - Flag to whether to use model picker or not.
        '''
        self._agent_name_ = agent_name
        self._deepracer_checkpoint_json = deepracer_checkpoint_json
        self._s3_metrics = Metrics(bucket=s3_dict_metrics[MetricsS3Keys.METRICS_BUCKET.value],
                                   s3_key=s3_dict_metrics[MetricsS3Keys.METRICS_KEY.value],
                                   region_name=s3_dict_metrics[MetricsS3Keys.REGION.value],
                                   s3_endpoint_url=s3_dict_metrics[MetricsS3Keys.ENDPOINT_URL.value])
        self._start_time_ = time.time()
        self._episode_ = 0
        self._episode_reward_ = 0.0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        self._is_eval_ = True
        self._eval_trials_ = 0
        self._checkpoint_state_ = CheckpointStateFile(ckpnt_dir)
        self._use_model_picker = use_model_picker
        self._eval_stats_dict_ = {'chkpnt_name': None, 'avg_comp_pct': -1.0}
        self._best_chkpnt_stats = {'name': None, 'avg_comp_pct': -1.0, 'time_stamp': time.time()}
        self._current_eval_pct_list_ = list()
        self.is_save_simtrace_enabled = rospy.get_param('SIMTRACE_S3_BUCKET', None)
        self.track_data = TrackData.get_instance()
        run_phase_sink.register(self)
        # Create the agent specific directories needed for storing the metric files
        self._simtrace_local_path = SIMTRACE_TRAINING_LOCAL_PATH_FORMAT.format(self._agent_name_)
        simtrace_dirname = os.path.dirname(self._simtrace_local_path)
        if simtrace_dirname or not os.path.exists(simtrace_dirname):
            os.makedirs(simtrace_dirname)
        self._current_sim_time = 0
        rospy.Service("/{}/{}".format(self._agent_name_, "mp4_video_metrics"), VideoMetricsSrv,
                      self._handle_get_video_metrics)
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        AbstractTracker.__init__(self, TrackerPriority.HIGH)

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        self._current_sim_time = sim_time.clock.secs + 1.e-9 * sim_time.clock.nsecs

    def reset(self):
        self._start_time_ = self._current_sim_time
        self._episode_reward_ = 0.0
        self._progress_ = 0.0

    def append_episode_metrics(self):
        self._episode_ += 1 if not self._is_eval_ else 0
        self._eval_trials_ += 1 if not self._is_eval_ else 0
        training_metric = dict()
        training_metric['reward_score'] = int(round(self._episode_reward_))
        training_metric['metric_time'] = int(round(self._current_sim_time * 1000))
        training_metric['start_time'] = int(round(self._start_time_ * 1000))
        training_metric['elapsed_time_in_milliseconds'] = \
            int(round((self._current_sim_time - self._start_time_) * 1000))
        training_metric['episode'] = int(self._episode_)
        training_metric['trial'] = int(self._eval_trials_)
        training_metric['phase'] = 'evaluation' if self._is_eval_ else 'training'
        training_metric['completion_percentage'] = int(self._progress_)
        training_metric['episode_status'] = EpisodeStatus.get_episode_status_label(self._episode_status)
        self._metrics_.append(training_metric)

    def upload_episode_metrics(self):
        json_metrics = json.dumps({'metrics': self._metrics_})
        self._s3_metrics.persist(body=json_metrics,
                                 s3_kms_extra_args=get_s3_kms_extra_args())
        if self._is_eval_:
            self._current_eval_pct_list_.append(self._progress_)

    def upload_step_metrics(self, metrics):
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        if not self._is_eval_:
            metrics[StepMetrics.EPISODE.value] = self._episode_
            self._episode_reward_ += metrics[StepMetrics.REWARD.value]
            StepMetrics.validate_dict(metrics)
            sim_trace_log(metrics)
            if self.is_save_simtrace_enabled:
                write_simtrace_to_local_file(self._simtrace_local_path,
                                             metrics)
        self._update_mp4_video_metrics(metrics)

    def update(self, data):
        self._is_eval_ = data != RunPhase.TRAIN

        if not self._is_eval_ and self._use_model_picker:
            if self._eval_stats_dict_['chkpnt_name'] is None:
                self._eval_stats_dict_['chkpnt_name'] = self._checkpoint_state_.read().name

            self._eval_trials_ = 0
            mean_pct = statistics.mean(self._current_eval_pct_list_ if \
                                       self._current_eval_pct_list_ else [0.0])
            LOGGER.info('Number of evaluations: {} Evaluation progresses: {}'.format(len(self._current_eval_pct_list_),
                                                                                     self._current_eval_pct_list_))
            LOGGER.info('Evaluation progresses mean: {}'.format(mean_pct))
            self._current_eval_pct_list_.clear()

            time_stamp = self._current_sim_time
            if mean_pct >= self._eval_stats_dict_['avg_comp_pct']:
                LOGGER.info('Current mean: {} >= Current best mean: {}'.format(mean_pct,
                                                                               self._eval_stats_dict_['avg_comp_pct']))
                LOGGER.info('Updating the best checkpoint to "{}" from "{}".'.format(self._eval_stats_dict_['chkpnt_name'],
                                                                                     self._best_chkpnt_stats['name']))
                self._eval_stats_dict_['avg_comp_pct'] = mean_pct
                self._best_chkpnt_stats = {'name': self._eval_stats_dict_['chkpnt_name'],
                                           'avg_comp_pct': mean_pct,
                                           'time_stamp': time_stamp}
            last_chkpnt_stats = {'name': self._eval_stats_dict_['chkpnt_name'],
                                 'avg_comp_pct': mean_pct,
                                 'time_stamp': time_stamp}
            self._deepracer_checkpoint_json.persist(
                body=json.dumps({BEST_CHECKPOINT: self._best_chkpnt_stats,
                                 LAST_CHECKPOINT: last_chkpnt_stats}),
                s3_kms_extra_args=get_s3_kms_extra_args())
            # Update the checkpoint name to the new checkpoint being used for training that will
            # then be evaluated, note this class gets notfied when the system is put into a
            # training phase and assumes that a training phase only starts when a new check point
            # is avaialble
            self._eval_stats_dict_['chkpnt_name'] = self._checkpoint_state_.read().name

    def _update_mp4_video_metrics(self, metrics):
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

        object_poses = [pose for object_name, pose in self.track_data.object_poses.items()\
                        if not object_name.startswith('racecar')]
        object_locations = []
        for pose in object_poses:
            point = Point32()
            point.x, point.y, point.z = pose.position.x, pose.position.y, 0
            object_locations.append(point)
        self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value] = object_locations

    def _handle_get_video_metrics(self, req):
        return VideoMetricsSrvResponse(self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value],
                                       self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.THROTTLE.value],
                                       self._video_metrics[Mp4VideoMetrics.STEERING.value],
                                       self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.DONE.value],
                                       self._video_metrics[Mp4VideoMetrics.X.value],
                                       self._video_metrics[Mp4VideoMetrics.Y.value],
                                       self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value])

class EvalMetrics(MetricsInterface, AbstractTracker):
    '''This class is responsible for uploading eval metrics to s3'''
    def __init__(self, agent_name, s3_dict_metrics, is_continuous):
        '''Init eval metrics

        Args:
            agent_name (string): agent name
            s3_dict_metrics (dict): Dictionary containing the required
                s3 info for the metrics bucket with keys specified by MetricsS3Keys
            is_continuous (bool): True if continuous race, False otherwise
        '''
        self._agent_name_ = agent_name
        self._s3_metrics = Metrics(bucket=s3_dict_metrics[MetricsS3Keys.METRICS_BUCKET.value],
                                   s3_key=s3_dict_metrics[MetricsS3Keys.METRICS_KEY.value],
                                   region_name=s3_dict_metrics[MetricsS3Keys.REGION.value],
                                   s3_endpoint_url=s3_dict_metrics[MetricsS3Keys.ENDPOINT_URL.value])
        self._is_continuous = is_continuous
        self._start_time_ = time.time()
        self._number_of_trials_ = 0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        # This is used to calculate the actual distance traveled by the car
        self._agent_xy = list()
        self._prev_step_time = time.time()
        self.is_save_simtrace_enabled = rospy.get_param('SIMTRACE_S3_BUCKET', None)
        # Create the agent specific directories needed for storing the metric files
        self._simtrace_local_path = SIMTRACE_EVAL_LOCAL_PATH_FORMAT.format(self._agent_name_)
        simtrace_dirname = os.path.dirname(self._simtrace_local_path)
        if simtrace_dirname or not os.path.exists(simtrace_dirname):
            os.makedirs(simtrace_dirname)
        self.reset_count_dict = {EpisodeStatus.CRASHED.value: 0,
                                 EpisodeStatus.OFF_TRACK.value: 0,
                                 EpisodeStatus.IMMOBILIZED.value: 0,
                                 EpisodeStatus.REVERSED.value: 0}
        self._best_lap_time = float('inf')
        self._total_evaluation_time = 0
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        self._reset_count_sum = 0
        self._current_sim_time = 0
        self.track_data = TrackData.get_instance()
        rospy.Service("/{}/{}".format(self._agent_name_, "mp4_video_metrics"), VideoMetricsSrv,
                      self._handle_get_video_metrics)
        AbstractTracker.__init__(self, TrackerPriority.HIGH)

    def update_tracker(self, delta_time, sim_time):
        """
        Callback when sim time is updated

        Args:
            delta_time (float): time diff from last call
            sim_time (Clock): simulation time
        """
        self._current_sim_time = sim_time.clock.secs + 1.e-9 * sim_time.clock.nsecs

    def reset(self):
        self._start_time_ = self._current_sim_time
        self._reset_count_sum += \
            self.reset_count_dict[EpisodeStatus.CRASHED.value] +\
            self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value] +\
            self.reset_count_dict[EpisodeStatus.OFF_TRACK.value] +\
            self.reset_count_dict[EpisodeStatus.REVERSED.value]
        for key in self.reset_count_dict.keys():
            self.reset_count_dict[key] = 0

    def append_episode_metrics(self):
        self._number_of_trials_ += 1
        eval_metric = dict()
        eval_metric['completion_percentage'] = int(self._progress_)
        eval_metric['metric_time'] = int(round(self._current_sim_time * 1000))
        eval_metric['start_time'] = int(round(self._start_time_ * 1000))
        eval_metric['elapsed_time_in_milliseconds'] = \
            int(round((self._current_sim_time - self._start_time_) * 1000))
        eval_metric['trial'] = int(self._number_of_trials_)
        eval_metric['episode_status'] = EpisodeStatus.get_episode_status_label(self._episode_status)
        eval_metric['crash_count'] = self.reset_count_dict[EpisodeStatus.CRASHED.value]
        eval_metric['immobilized_count'] = self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value]
        eval_metric['off_track_count'] = self.reset_count_dict[EpisodeStatus.OFF_TRACK.value]
        eval_metric['reversed_count'] = self.reset_count_dict[EpisodeStatus.REVERSED.value]
        eval_metric['reset_count'] = eval_metric['crash_count'] + \
                                     eval_metric['immobilized_count'] + \
                                     eval_metric['off_track_count'] + \
                                     eval_metric['reversed_count']
        self._best_lap_time = min(eval_metric['elapsed_time_in_milliseconds'], self._best_lap_time)
        self._total_evaluation_time += eval_metric['elapsed_time_in_milliseconds']
        self._metrics_.append(eval_metric)

    def upload_episode_metrics(self):
        json_metrics = json.dumps({'metrics': self._metrics_})
        self._s3_metrics.persist(body=json_metrics,
                                 s3_kms_extra_args=get_s3_kms_extra_args())

    def _update_mp4_video_metrics(self, metrics):
        actual_speed = 0
        cur_time = self._current_sim_time
        agent_x, agent_y = metrics[StepMetrics.X.value], metrics[StepMetrics.Y.value]
        if self._agent_xy:
            # Speed = Distance/Time
            delta_time = cur_time - self._prev_step_time
            actual_speed = 0
            if delta_time:
                actual_speed = math.sqrt((self._agent_xy[0] - agent_x) ** 2 +
                                         (self._agent_xy[1] - agent_y) ** 2) / delta_time
        self._agent_xy = [agent_x, agent_y]
        self._prev_step_time = cur_time

        self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value] = self._number_of_trials_
        self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value] = self._progress_
        # For continuous race, MP4 video will display the total reset counter for the entire race
        # For non-continuous race, MP4 video will display reset counter per lap
        self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value] = \
            self.reset_count_dict[EpisodeStatus.CRASHED.value] + \
            self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value] + \
            self.reset_count_dict[EpisodeStatus.OFF_TRACK.value] + \
            self.reset_count_dict[EpisodeStatus.REVERSED.value] + \
            (self._reset_count_sum if self._is_continuous else 0)

        self._video_metrics[Mp4VideoMetrics.THROTTLE.value] = actual_speed
        self._video_metrics[Mp4VideoMetrics.STEERING.value] = metrics[StepMetrics.STEER.value]
        self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value] = self._best_lap_time
        self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value] = self._total_evaluation_time +\
                                int(round((self._current_sim_time - self._start_time_) * 1000))
        self._video_metrics[Mp4VideoMetrics.DONE.value] = metrics[StepMetrics.DONE.value]
        self._video_metrics[Mp4VideoMetrics.X.value] = agent_x
        self._video_metrics[Mp4VideoMetrics.Y.value] = agent_y

        object_poses = [pose for object_name, pose in self.track_data.object_poses.items()\
                        if not object_name.startswith('racecar')]
        object_locations = []
        for pose in object_poses:
            point = Point32()
            point.x, point.y, point.z = pose.position.x, pose.position.y, 0
            object_locations.append(point)
        self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value] = object_locations

    def upload_step_metrics(self, metrics):
        metrics[StepMetrics.EPISODE.value] = self._number_of_trials_
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        if self._episode_status in self.reset_count_dict:
            self.reset_count_dict[self._episode_status] += 1
        StepMetrics.validate_dict(metrics)
        sim_trace_log(metrics)
        if self.is_save_simtrace_enabled:
            write_simtrace_to_local_file(self._simtrace_local_path,
                                         metrics)
        self._update_mp4_video_metrics(metrics)

    def _handle_get_video_metrics(self, req):
        return VideoMetricsSrvResponse(self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value],
                                       self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.THROTTLE.value],
                                       self._video_metrics[Mp4VideoMetrics.STEERING.value],
                                       self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.DONE.value],
                                       self._video_metrics[Mp4VideoMetrics.X.value],
                                       self._video_metrics[Mp4VideoMetrics.Y.value],
                                       self._video_metrics[Mp4VideoMetrics.OBJECT_LOCATIONS.value])
