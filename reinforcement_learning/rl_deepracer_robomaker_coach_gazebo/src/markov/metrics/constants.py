'''This module houses all the constants for the metrics package'''
from enum import Enum
from collections import OrderedDict

class MetricsS3Keys(Enum):
    '''The keys fro the s3 buckets'''
    REGION = 'aws_region'
    METRICS_BUCKET = 'metrics_bucket'
    METRICS_KEY = 'metrics_key'
    STEP_BUCKET = 'step_bucket'
    STEP_KEY = 'step_key'

class EvalMetricsKeys(Enum):
    '''The shared metric key for eval metrics'''
    PROGRESS = 'progress'

class StepMetrics(Enum):
    '''The keys for the sim trace metrics'''
    EPISODE = 'episode'
    STEPS = 'steps'
    X = 'X'
    Y = 'Y'
    YAW = 'yaw'
    STEER = 'steer'
    THROTTLE = 'throttle'
    ACTION = 'action'
    REWARD = 'reward'
    DONE = 'done'
    WHEELS_TRACK = 'all_wheels_on_track'
    PROG = 'progress'
    CLS_WAYPNT = 'closest_waypoint'
    TRACK_LEN = 'track_len'
    TIME = 'tstamp'
    EPISODE_STATUS = 'episode_status'

    @classmethod
    def make_default_metric(cls):
        '''Returns the default step metrics dict'''
        step_metrics = OrderedDict()
        for key in cls:
            step_metrics[key.value] = None
        return step_metrics

    @classmethod
    def validate_dict(cls, imput_dict):
        '''Throws an exception if a key is missing'''
        for key in cls:
            _= imput_dict[key.value]


class EpisodeStatus(Enum):
    '''The keys for episode status'''
    LAP_COMPLETE = 'lap_complete'
    CRASHED = 'crashed'
    OFF_TRACK = 'off_track'
    IN_PROGRESS = 'in_progress'
    IMMOBILIZED = 'immobilized'

    @classmethod
    def get_episode_status(cls, is_crashed, is_immobilized, is_off_track, is_lap_complete):
        episode_status = EpisodeStatus.IN_PROGRESS
        if is_off_track:
            episode_status = EpisodeStatus.OFF_TRACK
        elif is_crashed:
            episode_status = EpisodeStatus.CRASHED
        elif is_immobilized:
            episode_status = EpisodeStatus.IMMOBILIZED
        elif is_lap_complete:
            episode_status = EpisodeStatus.LAP_COMPLETE
        return episode_status

    @classmethod
    def get_episode_status_label(cls, episode_status):
        if isinstance(episode_status, str):
            return EPISODE_STATUS_LABEL_MAP[episode_status]
        elif isinstance(episode_status, EpisodeStatus):
            return EPISODE_STATUS_LABEL_MAP[episode_status.value]
        else:
            return EPISODE_STATUS_LABEL_MAP[str(episode_status)]


EPISODE_STATUS_LABEL_MAP = {
    EpisodeStatus.LAP_COMPLETE.value: 'Lap complete',
    EpisodeStatus.CRASHED.value: 'Crashed',
    EpisodeStatus.OFF_TRACK.value: 'Off track',
    EpisodeStatus.IMMOBILIZED.value: 'Immobilized',
    EpisodeStatus.IN_PROGRESS.value: 'In progress'
}
