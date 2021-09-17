"""This module contains all virtual event related constants"""
from enum import Enum

RACER_INFO_OBJECT = "RacerInformation"
RACER_INFO_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "racerAlias": {"type": "string"},
        "carConfig": {
            "type": "object",
            "properties": {"carColor": {"type": "string"}},
            "required": ["carColor"],
        },
        "inputModel": {
            "type": "object",
            "properties": {
                "s3BucketName": {"type": "string"},
                "s3KeyPrefix": {"type": "string"},
                "s3KmsKeyArn": {"type": "string"},
            },
            "required": ["s3BucketName", "s3KeyPrefix"],
        },
        "outputMetrics": {
            "type": "object",
            "properties": {
                "s3BucketName": {"type": "string"},
                "s3KeyPrefix": {"type": "string"},
                "s3KmsKeyArn": {"type": "string"},
            },
            "required": ["s3BucketName", "s3KeyPrefix"],
        },
        "outputStatus": {
            "type": "object",
            "properties": {
                "s3BucketName": {"type": "string"},
                "s3KeyPrefix": {"type": "string"},
                "s3KmsKeyArn": {"type": "string"},
            },
            "required": ["s3BucketName", "s3KeyPrefix"],
        },
        "outputSimTrace": {
            "type": "object",
            "properties": {
                "s3BucketName": {"type": "string"},
                "s3KeyPrefix": {"type": "string"},
                "s3KmsKeyArn": {"type": "string"},
            },
            "required": ["s3BucketName", "s3KeyPrefix"],
        },
        "outputMp4": {
            "type": "object",
            "properties": {
                "s3BucketName": {"type": "string"},
                "s3KeyPrefix": {"type": "string"},
                "s3KmsKeyArn": {"type": "string"},
            },
            "required": ["s3BucketName", "s3KeyPrefix"],
        },
    },
    "required": [
        "racerAlias",
        "carConfig",
        "inputModel",
        "outputMetrics",
        "outputStatus",
        "outputSimTrace",
        "outputMp4",
    ],
}
CAR_CONTROL_INPUT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "object",
    "properties": {
        "type": {"type": "string"},
        "payload": {
            "type": "object",
            "properties": {
                "statusMode": {"type": "string"},
                "speedMode": {"type": "string"},
                "speedValue": {"type": "string"},
            },
            "anyOf": [{"required": ["statusMode"]}, {"required": ["speedMode", "speedValue"]}],
        },
        "sentTime": {"type": "string"},
    },
    "required": ["type", "payload", "sentTime"],
}

SENSOR_MODEL_MAP = {
    "single_camera": "racecar_0",
    "stereo_camera": "racecar_1",
    "single_camera_lidar": "racecar_2",
    "stereo_camera_lidar": "racecar_3",
}
MAX_NUM_OF_SQS_MESSAGE = 1
SQS_WAIT_TIME_SEC = 5
MAX_NUM_OF_SQS_ERROR = 10
DEFAULT_RACE_DURATION = 180  # default race duration of 180 seconds
VIRTUAL_EVENT = "virtual_event"
# second to pause before the start of the race
PAUSE_TIME_BEFORE_START = 10.0

# pause second after race finish to show finish icon
PAUSE_TIME_AFTER_FINISH = 2.0

# wait state constant value
# TODO: figure out what display name to use during wait
WAIT_DISPLAY_NAME = "TBD"
WAIT_CURRENT_LAP = 1
WAIT_TOTAL_EVAL_SECONDS = 0
WAIT_RESET_COUNTER = 0
WAIT_SPEED = 0

WEBRTC_DATA_PUB_TOPIC = "/string/raw"
WEBRTC_CAR_CTRL_FORMAT = "/webrtc/{}"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
MAX_SPEED = 4  # 4 m/s
MIN_SPEED = 0.1  # 0.1 m/s


class CarControlTopic(Enum):
    """Keys for WebRTC messages used for car control"""

    STATUS_CTRL = "status_ctrl"
    SPEED_CTRL = "speed_ctrl"


class WebRTCCarControl(Enum):
    """Keys for WebRTC messages used for car control"""

    TYPE = "type"
    PAYLOAD = "payload"
    STATUS = "status"
    SPEED = "speed"
    SPEED_MODE = "speedMode"
    SPEED_VALUE = "speedValue"
    STATUS_MODE = "statusMode"
    SENT_TIME = "sentTime"


class CarControlStatus(Enum):
    """Status of webrtc based car control."""

    RESUME = "resume"
    PAUSE = "pause"


class CarControlMode(Enum):
    """For we have 5 control modes so far.

    abs: is passing absolution speed that entirely overrides the model's speed
        e.g. input 2 m/s or input 0.5 m/s
    multiplier: is percentage of the current model speed
        e.g. current model speed is 2 m/s, input can be [0, inf]
        input 0.5 => 0.5 * 2 m/s = 1 m/s
        input 1.5 => 1.5 * 2 m/s = 3 m/s
        input 100 => 1.5 * 2 m/s = 4 m/s (max speed)
    percent_max: is percentage of the current model speed
        e.g. max speed is 4 m/s, input can be [0, 1]
        input 0.5 => 0.5 * 4 m/s = 2 m/s
        input 1.5 => 1.5 * 4 m/s = 4 m/s
    offset: is adding speed on top of current model speed
        e.g. max speed is 8 m/s, input need to be [-inf, inf]
        input -0.5 => 2 m/s - 0.5 m/s= 1.5 m/s
        input 0.5 => 2 m/s + 0.5 m/s = 2.5 m/s
    model_speed: the speed outputed by the model we trained.
    """

    ABSOLUTE = "abs"
    MULTIPLIER = "multiplier"
    PERCENT_MAX = "percent_max"
    OFFSET = "offset"
    MODEL_SPEED = "model_speed"
