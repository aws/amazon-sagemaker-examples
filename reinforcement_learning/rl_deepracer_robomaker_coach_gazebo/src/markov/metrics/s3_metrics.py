'''This module houses the metric objects for the sim app'''
import time
import json
import logging
import boto3
from markov import utils
from markov.metrics.constants import MetricsS3Keys, StepMetrics, EpisodeStatus
from markov.metrics.metrics_interface import MetricsInterface
from markov.s3_simdata_upload import DeepRacerRacetrackSimTraceData

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
    session = boto3.session.Session()
    s3_client = session.client('s3', region_name=region)
    s3_client.put_object(Bucket=bucket, Key=key,
                         Body=bytes(json.dumps({'metrics': metrics}),
                                    encoding='utf-8'))

class TrainingMetrics(MetricsInterface):
    '''This class is responsible for uploading training metrics to s3'''
    def __init__(self, s3_dict):
        '''s3_dict - Dictionary containing the required s3 info with keys
                     specified by MetricsS3Keys
        '''
        self._s3_dict_ = s3_dict
        self._start_time_ = time.time()
        self._episode_ = 0
        self._episode_reward_ = 0.0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        self._simtrace_data_ = \
            DeepRacerRacetrackSimTraceData(self._s3_dict_[MetricsS3Keys.STEP_BUCKET.value],
                                           self._s3_dict_[MetricsS3Keys.STEP_KEY.value])

    def reset(self):
        self._start_time_ = time.time()
        self._episode_reward_ = 0.0
        self._progress_ = 0.0

    def upload_episode_metrics(self):
        self._episode_ += 1
        training_metric = dict()
        training_metric['reward_score'] = int(round(self._episode_reward_))
        training_metric['metric_time'] = int(round(time.time() * 1000))
        training_metric['start_time'] = int(round(self._start_time_ * 1000))
        training_metric['elapsed_time_in_milliseconds'] = \
            int(round((time.time() - self._start_time_) * 1000))
        training_metric['episode'] = int(self._episode_)
        training_metric['completion_percentage'] = int(self._progress_)
        training_metric['episode_status'] = EpisodeStatus.get_episode_status_label(self._episode_status)
        self._metrics_.append(training_metric)
        write_metrics_to_s3(self._s3_dict_[MetricsS3Keys.METRICS_BUCKET.value],
                            self._s3_dict_[MetricsS3Keys.METRICS_KEY.value],
                            self._s3_dict_[MetricsS3Keys.REGION.value],
                            self._metrics_)
        self._simtrace_data_.upload_to_s3(self._episode_)

    def upload_step_metrics(self, metrics):
        metrics[StepMetrics.EPISODE.value] = self._episode_
        self._episode_reward_ += metrics[StepMetrics.REWARD.value]
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        sim_trace_log(metrics)
        self._simtrace_data_.write_simtrace_data(metrics)

class EvalMetrics(MetricsInterface):
    '''This class is responsible for uploading eval metrics to s3'''
    def __init__(self, s3_dict):
        '''s3_dict - Dictionary containing the required s3 info with keys
                     specified by MetricsS3Keys
        '''
        self._s3_dict_ = s3_dict
        self._start_time_ = time.time()
        self._number_of_trials_ = 0
        self._progress_ = 0.0
        self._episode_status = ''
        self._metrics_ = list()
        self._simtrace_data_ = \
            DeepRacerRacetrackSimTraceData(self._s3_dict_[MetricsS3Keys.STEP_BUCKET.value],
                                           self._s3_dict_[MetricsS3Keys.STEP_KEY.value])

    def reset(self):
        self._start_time_ = time.time()

    def upload_episode_metrics(self):
        self._number_of_trials_ += 1
        eval_metric = dict()
        eval_metric['completion_percentage'] = int(self._progress_)
        eval_metric['metric_time'] = int(round(time.time() * 1000))
        eval_metric['start_time'] = int(round(self._start_time_ * 1000))
        eval_metric['elapsed_time_in_milliseconds'] = \
            int(round((time.time() - self._start_time_) * 1000))
        eval_metric['trial'] = int(self._number_of_trials_)
        eval_metric['episode_status'] = EpisodeStatus.get_episode_status_label(self._episode_status)
        self._metrics_.append(eval_metric)
        write_metrics_to_s3(self._s3_dict_[MetricsS3Keys.METRICS_BUCKET.value],
                            self._s3_dict_[MetricsS3Keys.METRICS_KEY.value],
                            self._s3_dict_[MetricsS3Keys.REGION.value],
                            self._metrics_)
        self._simtrace_data_.upload_to_s3(self._number_of_trials_)

    def upload_step_metrics(self, metrics):
        metrics[StepMetrics.EPISODE.value] = self._number_of_trials_
        self._progress_ = metrics[StepMetrics.PROG.value]
        self._episode_status = metrics[StepMetrics.EPISODE_STATUS.value]
        sim_trace_log(metrics)
        self._simtrace_data_.write_simtrace_data(metrics)
