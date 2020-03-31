'''This module houses all the constants for the metrics package'''
from enum import Enum, unique
from collections import OrderedDict
import os
import logging
from markov.utils import Logger

LOG = Logger(__name__, logging.INFO).get_logger()

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
    def validate_dict(cls, input_dict):
        '''Throws an exception if a key is missing'''
        for key in cls:
            if input_dict[key.value] is None:
                raise Exception("StepMetrics dict's key({})'s value is None".format(key.value))


class EpisodeStatus(Enum):
    '''The keys for episode status'''
    EPISODE_COMPLETE = 'lap_complete'
    CRASHED = 'crashed'
    OFF_TRACK = 'off_track'
    IN_PROGRESS = 'in_progress'
    IMMOBILIZED = 'immobilized'
    PAUSE = 'pause'
    REVERSED = 'reversed'

    @classmethod
    def get_episode_status(cls, is_done_dict):
        # is_done_dict will have at most one True value or all False
        try:
            episode_status = list(is_done_dict.keys())[list(is_done_dict.values()).index(True)]
            return episode_status
        except ValueError:
            return EpisodeStatus.IN_PROGRESS.value

    @classmethod
    def get_episode_status_label(cls, episode_status):
        if isinstance(episode_status, str):
            return EPISODE_STATUS_LABEL_MAP[episode_status]
        elif isinstance(episode_status, EpisodeStatus):
            return EPISODE_STATUS_LABEL_MAP[episode_status.value]
        else:
            return EPISODE_STATUS_LABEL_MAP[str(episode_status)]


EPISODE_STATUS_LABEL_MAP = {
    EpisodeStatus.EPISODE_COMPLETE.value: 'Lap complete',
    EpisodeStatus.CRASHED.value: 'Crashed',
    EpisodeStatus.OFF_TRACK.value: 'Off track',
    EpisodeStatus.IMMOBILIZED.value: 'Immobilized',
    EpisodeStatus.IN_PROGRESS.value: 'In progress',
    EpisodeStatus.PAUSE.value: 'Pause',
    EpisodeStatus.REVERSED.value: 'Reversed'
}

#
# This constant is import in Markov and Simulation Application.
# This caused a race condition of checking for condition and trying to create folder.
# Hence having a try catch block. Python 3.2 has os.makedirs(mydir, exist_ok=True)
# will use this when we move to python3
#
try:
    ITERATION_DATA_LOCAL_FILE_PATH = "./custom_files/iteration_data/"
    if not os.path.exists(ITERATION_DATA_LOCAL_FILE_PATH):
        os.makedirs(ITERATION_DATA_LOCAL_FILE_PATH)
except OSError as err:
    LOG.info("The directory was already created")

class IterationDataLocalFileNames(Enum):
    ''' Local file names that should be uploaded to s3 after every rollout'''
    SIM_TRACE_TRAINING_LOCAL_FILE = 'training-simtrace/iteration.csv'
    SIM_TRACE_EVALUATION_LOCAL_FILE = 'evaluation-simtrace/iteration.csv'
    CAMERA_PIP_MP4_VALIDATION_LOCAL_PATH = 'camera-pip/video.mp4'
    CAMERA_45DEGREE_MP4_VALIDATION_LOCAL_PATH = 'camera-45degree/video.mp4'
    CAMERA_TOPVIEW_MP4_VALIDATION_LOCAL_PATH = 'camera-topview/video.mp4'


MULTIPROCESS_S3WRITER_POOL = 4

@unique
class Mp4VideoMetrics(Enum):
    '''This enum is used for gathering the video metrics displayed on the mp4'''
    LAP_COUNTER = 'lap_counter'
    COMPLETION_PERCENTAGE = 'completion_percentage'
    RESET_COUNTER = 'reset_counter'
    CRASH_COUNTER = 'crash_counter'
    THROTTLE = 'throttle'
    STEERING = 'steering'
    BEST_LAP_TIME = 'best_lap_time'
    TOTAL_EVALUATION_TIME = 'total_evaluation_time'
    DONE = 'done'

    @classmethod
    def get_empty_dict(cls):
        '''Returns dictionary with the string as key values and None's as the values, clients
           are responsible for populating the dict accordingly
        '''
        empty_dict = dict()
        for enum_map in cls._value2member_map_.values():
            empty_dict[enum_map.value] = None
        return empty_dict
