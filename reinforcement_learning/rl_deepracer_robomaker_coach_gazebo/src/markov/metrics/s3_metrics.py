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
from markov import utils
from markov.common import ObserverInterface
from markov.metrics.constants import (MetricsS3Keys, StepMetrics, EpisodeStatus,
                                      IterationDataLocalFileNames, ITERATION_DATA_LOCAL_FILE_PATH,
                                      Mp4VideoMetrics)
from markov.metrics.metrics_interface import MetricsInterface
from markov.s3_simdata_upload import DeepRacerRacetrackSimTraceData
from markov.utils import log_and_exit, get_boto_config, SIMAPP_SIMULATION_WORKER_EXCEPTION, \
                         SIMAPP_EVENT_ERROR_CODE_400, SIMAPP_EVENT_ERROR_CODE_500
from rl_coach.checkpoint import CheckpointStateFile
from rl_coach.core_types import RunPhase


LOGGER = utils.Logger(__name__, logging.INFO).get_logger()

#! TODO this needs to be removed after muti part is fixed, note we don't have
# agent name here, but we can add it to the step metrics if needed
def sim_trace_log(sim_trace_dict):
    '''Logs the step metrics to cloud watch
       sim_trace_dict - Ordered dict containing the step metrics, note order must match
                        precision in the string
    '''
    LOGGER.info('SIM_TRACE_LOG:%d,%d,%.4f,%.4f,%.4f,%.2f,%.2f,%d,%.4f,%s,%s,%.4f,%d,%.2f,%s,%s\n' % \
        (tuple(sim_trace_dict.values())))

def write_metrics_to_s3(bucket, key, region, metrics):
    '''Helper method that uploads the desired metrics to s3
       bucket - String with S3 bucket where metrics should be written
       key - String with S3 bucket key where metrics should be written
       region - String with aws region
       metrics - Dictionary with metrics to write to s3
    '''
    try:
        session = boto3.session.Session()
        s3_client = session.client('s3', region_name=region, config=get_boto_config())
        s3_client.put_object(Bucket=bucket,
                             Key=key,
                             Body=bytes(json.dumps(metrics), encoding='utf-8'))
    except botocore.exceptions.ClientError as err:
        log_and_exit("Unable to write metrics to s3: bucket: \
                      {}, error: {}".format(bucket, err.response['Error']['Code']),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_400)
    except Exception as ex:
        log_and_exit("Unable to write metrics to s3, exception: {}".format(ex),
                     SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)

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

class TrainingMetrics(MetricsInterface, ObserverInterface):
    '''This class is responsible for uploading training metrics to s3'''
    def __init__(self, agent_name, s3_dict_metrics, s3_dict_model, ckpnt_dir, run_phase_sink):
        '''s3_dict_metrics - Dictionary containing the required s3 info for the metrics
                             bucket with keys specified by MetricsS3Keys
           s3_dict_model - Dictionary containing the required s3 info for the model
                           bucket, which is where the best model info will be saved with
                           keys specified by MetricsS3Keys
           ckpnt_dir - Directory where the current checkpont is to be stored
           run_phase_sink - Sink to recieve notification of a change in run phase
        '''
        self._agent_name_ = agent_name
        self._s3_dict_metrics_ = s3_dict_metrics
        self._s3_dict_model_ = s3_dict_model
        self._start_time_ = time.time()
        self._episode_ = 0
        self._episode_reward_ = 0.0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        self._is_eval_ = True
        self._eval_trials_ = 0
        self._checkpoint_state_ = CheckpointStateFile(ckpnt_dir)
        self._eval_stats_dict_ = {'chkpnt_name': None, 'avg_comp_pct': 0.0}
        self._best_chkpnt_stats = {'name': None, 'avg_comp_pct': 0.0, 'time_stamp': time.time()}
        self._current_eval_pct_list_ = list()
        self._simtrace_data_ = \
            DeepRacerRacetrackSimTraceData(self._s3_dict_metrics_[MetricsS3Keys.STEP_BUCKET.value],
                                           self._s3_dict_metrics_[MetricsS3Keys.STEP_KEY.value])
        run_phase_sink.register(self)
        # Create the agent specific directories needed for storing the metric files
        simtrace_dirname = os.path.dirname(IterationDataLocalFileNames.SIM_TRACE_TRAINING_LOCAL_FILE.value)
        if not os.path.exists(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_, simtrace_dirname)):
            os.makedirs(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_, simtrace_dirname))

    def reset(self):
        self._start_time_ = time.time()
        self._episode_reward_ = 0.0
        self._progress_ = 0.0

    def append_episode_metrics(self):
        self._episode_ += 1 if not self._is_eval_ else 0
        self._eval_trials_ += 1 if not self._is_eval_ else 0
        training_metric = dict()
        training_metric['reward_score'] = int(round(self._episode_reward_))
        training_metric['metric_time'] = int(round(time.time() * 1000))
        training_metric['start_time'] = int(round(self._start_time_ * 1000))
        training_metric['elapsed_time_in_milliseconds'] = \
            int(round((time.time() - self._start_time_) * 1000))
        training_metric['episode'] = int(self._episode_)
        training_metric['trial'] = int(self._eval_trials_)
        training_metric['phase'] = 'evaluation' if self._is_eval_ else 'training'
        training_metric['completion_percentage'] = int(self._progress_)
        training_metric['episode_status'] = EpisodeStatus.get_episode_status_label(self._episode_status)
        self._metrics_.append(training_metric)

    def upload_episode_metrics(self):
        write_metrics_to_s3(self._s3_dict_metrics_[MetricsS3Keys.METRICS_BUCKET.value],
                            self._s3_dict_metrics_[MetricsS3Keys.METRICS_KEY.value],
                            self._s3_dict_metrics_[MetricsS3Keys.REGION.value],
                            {'metrics': self._metrics_})
        if self._is_eval_:
            self._current_eval_pct_list_.append(self._progress_)

    def upload_step_metrics(self, metrics):
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        self._episode_reward_ += metrics[StepMetrics.REWARD.value]
        #! TODO have this work with new sim trace class
        if not self._is_eval_:
            metrics[StepMetrics.EPISODE.value] = self._episode_
            self._episode_reward_ += metrics[StepMetrics.REWARD.value]
            StepMetrics.validate_dict(metrics)
            sim_trace_log(metrics)
            is_save_simtrace_enabled = rospy.get_param('SIMTRACE_S3_BUCKET', None)
            if is_save_simtrace_enabled:
                write_simtrace_to_local_file(
                    os.path.join(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_),
                                 IterationDataLocalFileNames.SIM_TRACE_TRAINING_LOCAL_FILE.value),
                    metrics)
            self._simtrace_data_.write_simtrace_data(metrics)

    def update(self, data):
        self._is_eval_ = data != RunPhase.TRAIN

        if not self._is_eval_:
            if self._eval_stats_dict_['chkpnt_name'] is None:
                self._eval_stats_dict_['chkpnt_name'] = self._checkpoint_state_.read().name

            self._eval_trials_ = 0
            mean_pct = statistics.mean(self._current_eval_pct_list_ if \
                                       self._current_eval_pct_list_ else [-1])
            self._current_eval_pct_list_.clear()

            time_stamp = time.time()
            if mean_pct >= self._eval_stats_dict_['avg_comp_pct']:
                self._eval_stats_dict_['avg_comp_pct'] = mean_pct
                self._best_chkpnt_stats = {'name': self._eval_stats_dict_['chkpnt_name'],
                                           'avg_comp_pct': mean_pct,
                                           'time_stamp': time_stamp}
            last_chkpnt_stats = {'name': self._eval_stats_dict_['chkpnt_name'],
                                 'avg_comp_pct': mean_pct,
                                 'time_stamp': time_stamp}
            write_metrics_to_s3(self._s3_dict_model_[MetricsS3Keys.METRICS_BUCKET.value],
                                self._s3_dict_model_[MetricsS3Keys.METRICS_KEY.value],
                                self._s3_dict_model_[MetricsS3Keys.REGION.value],
                                {utils.BEST_CHECKPOINT: self._best_chkpnt_stats,
                                 utils.LAST_CHECKPOINT: last_chkpnt_stats})
            # Update the checkpoint name to the new checkpoint being used for training that will
            # then be evaluated, note this class gets notfied when the system is put into a
            # training phase and assumes that a training phase only starts when a new check point
            # is avaialble
            self._eval_stats_dict_['chkpnt_name'] = self._checkpoint_state_.read().name

class EvalMetrics(MetricsInterface):
    '''This class is responsible for uploading eval metrics to s3'''
    def __init__(self, agent_name, s3_dict_metrics):
        '''s3_dict_metrics - Dictionary containing the required s3 info for the metrics
                             bucket with keys specified by MetricsS3Keys
        '''
        self._agent_name_ = agent_name
        self._s3_dict_metrics_ = s3_dict_metrics
        self._start_time_ = time.time()
        self._number_of_trials_ = 0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        # This is used to calculate the actual distance travelled by the car
        self._agent_xy = list()
        self._prev_step_time = None
        self._simtrace_data_ = \
            DeepRacerRacetrackSimTraceData(self._s3_dict_metrics_[MetricsS3Keys.STEP_BUCKET.value],
                                           self._s3_dict_metrics_[MetricsS3Keys.STEP_KEY.value])
        # Create the agent specific directories needed for storing the metric files
        simtrace_dirname = os.path.dirname(IterationDataLocalFileNames.SIM_TRACE_EVALUATION_LOCAL_FILE.value)
        if not os.path.exists(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_, simtrace_dirname)):
            os.makedirs(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_, simtrace_dirname))
        self.reset_count_dict = {EpisodeStatus.CRASHED.value: 0,
                                 EpisodeStatus.OFF_TRACK.value: 0,
                                 EpisodeStatus.IMMOBILIZED.value: 0,
                                 EpisodeStatus.REVERSED.value: 0}
        self._best_lap_time = float('inf')
        self._total_evaluation_time = 0
        self._video_metrics = Mp4VideoMetrics.get_empty_dict()
        rospy.Service("/{}/{}".format(self._agent_name_, "mp4_video_metrics"), VideoMetricsSrv,
                      self._handle_get_video_metrics)

    def reset(self):
        self._start_time_ = time.time()
        for key in self.reset_count_dict.keys():
            self.reset_count_dict[key] = 0

    def append_episode_metrics(self):
        self._number_of_trials_ += 1
        eval_metric = dict()
        eval_metric['completion_percentage'] = int(self._progress_)
        eval_metric['metric_time'] = int(round(time.time() * 1000))
        eval_metric['start_time'] = int(round(self._start_time_ * 1000))
        eval_metric['elapsed_time_in_milliseconds'] = \
            int(round((time.time() - self._start_time_) * 1000))
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
        write_metrics_to_s3(self._s3_dict_metrics_[MetricsS3Keys.METRICS_BUCKET.value],
                            self._s3_dict_metrics_[MetricsS3Keys.METRICS_KEY.value],
                            self._s3_dict_metrics_[MetricsS3Keys.REGION.value],
                            {'metrics': self._metrics_})

    def _update_mp4_video_metrics(self, metrics):
        actual_speed = 0
        cur_time = time.time()
        agent_x, agent_y = metrics[StepMetrics.X.value], metrics[StepMetrics.Y.value]
        if self._agent_xy:
            # Speed = Distance/Time
            delta_time = cur_time - self._prev_step_time
            actual_speed = math.sqrt((self._agent_xy[0] - agent_x) ** 2 +
                                     (self._agent_xy[1] - agent_y) ** 2) / delta_time
        self._agent_xy = [agent_x, agent_y]
        self._prev_step_time = cur_time

        self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value] = self._number_of_trials_
        self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value] = self._progress_
        self._video_metrics[Mp4VideoMetrics.CRASH_COUNTER.value] = self.reset_count_dict[EpisodeStatus.CRASHED.value]
        self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value] = \
            self.reset_count_dict[EpisodeStatus.CRASHED.value] +\
            self.reset_count_dict[EpisodeStatus.IMMOBILIZED.value] +\
            self.reset_count_dict[EpisodeStatus.OFF_TRACK.value] +\
            self.reset_count_dict[EpisodeStatus.REVERSED.value]

        self._video_metrics[Mp4VideoMetrics.THROTTLE.value] = actual_speed
        self._video_metrics[Mp4VideoMetrics.STEERING.value] = metrics[StepMetrics.STEER.value]
        self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value] = self._best_lap_time
        self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value] = self._total_evaluation_time +\
                                int(round((time.time() - self._start_time_) * 1000))
        self._video_metrics[Mp4VideoMetrics.DONE.value] = metrics[StepMetrics.DONE.value]

    def upload_step_metrics(self, metrics):
        metrics[StepMetrics.EPISODE.value] = self._number_of_trials_
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        if self._episode_status in self.reset_count_dict:
            self.reset_count_dict[self._episode_status] += 1
        StepMetrics.validate_dict(metrics)
        sim_trace_log(metrics)
        is_save_simtrace_enabled = rospy.get_param('SIMTRACE_S3_BUCKET', None)
        if is_save_simtrace_enabled:
            write_simtrace_to_local_file(
                os.path.join(os.path.join(ITERATION_DATA_LOCAL_FILE_PATH, self._agent_name_),
                             IterationDataLocalFileNames.SIM_TRACE_EVALUATION_LOCAL_FILE.value),
                metrics)
        self._simtrace_data_.write_simtrace_data(metrics)
        self._update_mp4_video_metrics(metrics)

    def _handle_get_video_metrics(self, req):
        return VideoMetricsSrvResponse(self._video_metrics[Mp4VideoMetrics.LAP_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.COMPLETION_PERCENTAGE.value],
                                       self._video_metrics[Mp4VideoMetrics.RESET_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.CRASH_COUNTER.value],
                                       self._video_metrics[Mp4VideoMetrics.THROTTLE.value],
                                       self._video_metrics[Mp4VideoMetrics.STEERING.value],
                                       self._video_metrics[Mp4VideoMetrics.BEST_LAP_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.TOTAL_EVALUATION_TIME.value],
                                       self._video_metrics[Mp4VideoMetrics.DONE.value])
