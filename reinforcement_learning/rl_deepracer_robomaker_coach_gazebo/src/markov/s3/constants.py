'''This module contains all s3 related prefix, key, and local path constants'''

import os

from enum import Enum

# custom file path
CUSTOM_FILES_PATH = "./custom_files"

# F1 constants
F1_RACE_TYPE = "F1"
F1_SHELL_USERS_LIST = ["TataCalde", "RIC3", "SheBangsTheDrums1989"]

############################
# reward function download #
############################
REWARD_FUCTION_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                "{}/customer_reward_function.py")

##################################
# hyperparameter upload/download #
##################################
HYPERPARAMETER_S3_POSTFIX = "ip/hyperparameters.json"
HYPERPARAMETER_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                "{}/hyperparameters.json")

###########################
# model metadata download #
###########################
MODEL_METADATA_S3_POSTFIX = "model/model_metadata.json"
# for single agent, {} should be replaced by 'agent'
# for multi agent, {} should be replaced by 'agent_0' for the first agent
MODEL_METADATA_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                "{}/model_metadata.json")

#################
# yaml download #
#################
# replace {} with yaml file name
YAML_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH, "{}")


class AgentType(Enum):
    '''agent types for simapp'''
    ROLLOUT = "rollout"
    EVALUATION = "evaluation"
    TOURNAMENT = "tournament"


class YamlKey(Enum):
    '''yaml key for all types of workers'''
    RACE_TYPE_YAML_KEY = "RACE_TYPE"
    RACER_NAME_YAML_KEY = "RACER_NAME"
    VIDEO_JOB_TYPE_YAML_KEY = "VIDEO_JOB_TYPE"
    LEADERBOARD_TYPE_YAML_KEY = "LEADERBOARD_TYPE"
    LEADERBOARD_NAME_YAML_KEY = "LEADERBOARD_NAME"
    CAR_COLOR_YAML_KEY = "CAR_COLOR"
    MODEL_S3_BUCKET_YAML_KEY = "MODEL_S3_BUCKET"
    MODEL_S3_PREFIX_YAML_KEY = "MODEL_S3_PREFIX"
    METRICS_S3_BUCKET_YAML_KEY = "METRICS_S3_BUCKET"
    METRICS_S3_PREFIX_YAML_KEY = "METRICS_S3_PREFIX"
    METRICS_S3_OBJECT_KEY_YAML_KEY = "METRICS_S3_OBJECT_KEY"
    MODEL_METADATA_FILE_S3_YAML_KEY = "MODEL_METADATA_FILE_S3_KEY"
    BODY_SHELL_TYPE_YAML_KEY = "BODY_SHELL_TYPE"
    DISPLAY_NAME_YAML_KEY = "DISPLAY_NAME"
    SIMTRACE_S3_BUCKET_YAML_KEY = "SIMTRACE_S3_BUCKET"
    SIMTRACE_S3_PREFIX_YAML_KEY = "SIMTRACE_S3_PREFIX"
    MP4_S3_BUCKET_YAML_KEY = "MP4_S3_BUCKET"
    MP4_S3_PREFIX_YAML_KEY = "MP4_S3_OBJECT_PREFIX"
    SAGEMAKER_SHARED_S3_BUCKET_YAML_KEY = "SAGEMAKER_SHARED_S3_BUCKET"
    SAGEMAKER_SHARED_S3_PREFIX_YAML_KEY = "SAGEMAKER_SHARED_S3_PREFIX"


EVAL_MANDATORY_YAML_KEY = [YamlKey.MODEL_S3_BUCKET_YAML_KEY.value,
                           YamlKey.MODEL_S3_PREFIX_YAML_KEY.value,
                           YamlKey.METRICS_S3_BUCKET_YAML_KEY.value,
                           YamlKey.METRICS_S3_OBJECT_KEY_YAML_KEY.value,
                           YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value,
                           YamlKey.BODY_SHELL_TYPE_YAML_KEY.value,
                           YamlKey.CAR_COLOR_YAML_KEY.value]

TOUR_MANDATORY_YAML_KEY = [YamlKey.MODEL_S3_BUCKET_YAML_KEY.value,
                           YamlKey.MODEL_S3_PREFIX_YAML_KEY.value,
                           YamlKey.METRICS_S3_BUCKET_YAML_KEY.value,
                           YamlKey.METRICS_S3_PREFIX_YAML_KEY.value,
                           YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value,
                           YamlKey.BODY_SHELL_TYPE_YAML_KEY.value,
                           YamlKey.DISPLAY_NAME_YAML_KEY.value,
                           YamlKey.SIMTRACE_S3_BUCKET_YAML_KEY.value,
                           YamlKey.SIMTRACE_S3_PREFIX_YAML_KEY.value,
                           YamlKey.MP4_S3_BUCKET_YAML_KEY.value,
                           YamlKey.MP4_S3_PREFIX_YAML_KEY.value]

TRAINING_MANDATORY_YAML_KEY = [YamlKey.SAGEMAKER_SHARED_S3_BUCKET_YAML_KEY.value,
                               YamlKey.SAGEMAKER_SHARED_S3_PREFIX_YAML_KEY.value,
                               YamlKey.METRICS_S3_BUCKET_YAML_KEY.value,
                               YamlKey.METRICS_S3_OBJECT_KEY_YAML_KEY.value,
                               YamlKey.MODEL_METADATA_FILE_S3_YAML_KEY.value,
                               YamlKey.BODY_SHELL_TYPE_YAML_KEY.value,
                               YamlKey.CAR_COLOR_YAML_KEY.value]

#############################
# simtrace and video upload #
#############################
# {} should be replaced by member variable 
SIMTRACE_EVAL_S3_POSTFIX = "evaluation-simtrace/{}-iteration.csv"
SIMTRACE_TRAINING_S3_POSTFIX = "training-simtrace/{}-iteration.csv"
CAMERA_PIP_MP4_S3_POSTFIX = "camera-pip/{}-video.mp4"
CAMERA_45DEGREE_MP4_S3_POSTFIX = "camera-45degree/{}-video.mp4"
CAMERA_TOPVIEW_MP4_S3_POSTFIX = "camera-topview/{}-video.mp4"

# for single agent, {} should be replaced by 'agent'
# for multi agent, {} should be replaced by 'agent_0' for first agent
SIMTRACE_EVAL_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                               "iteration_data/{}/evaluation-simtrace/iteration.csv")
SIMTRACE_TRAINING_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                   "iteration_data/{}/training-simtrace/iteration.csv")
CAMERA_PIP_MP4_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                "iteration_data/{}/camera-pip/video.mp4")
CAMERA_45DEGREE_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                 "iteration_data/{}/camera-45degree/video.mp4")
CAMERA_TOPVIEW_LOCAL_PATH_FORMAT = os.path.join(CUSTOM_FILES_PATH,
                                                "iteration_data/{}/camera-topview/video.mp4")


class SimtraceVideoNames(Enum):
    SIMTRACE_EVAL = 'simtrace_eval'
    SIMTRACE_TRAINING = 'simtrace_training'
    PIP = 'pip'
    DEGREE45 = 'degree45'
    TOPVIEW = 'topview'


# simtrace video dict
SIMTRACE_VIDEO_POSTFIX_DICT = \
    {SimtraceVideoNames.SIMTRACE_EVAL.value: SIMTRACE_EVAL_S3_POSTFIX,
     SimtraceVideoNames.SIMTRACE_TRAINING.value: SIMTRACE_TRAINING_S3_POSTFIX,
     SimtraceVideoNames.PIP.value: CAMERA_PIP_MP4_S3_POSTFIX,
     SimtraceVideoNames.DEGREE45.value: CAMERA_45DEGREE_MP4_S3_POSTFIX,
     SimtraceVideoNames.TOPVIEW.value: CAMERA_TOPVIEW_MP4_S3_POSTFIX}

#############################
# ip config upload/download #
#############################
# 20 minutes
SAGEMAKER_WAIT_TIME = 1200

IP_ADDRESS_POSTFIX = "ip/ip.json"
IP_DONE_POSTFIX = "ip/done"

IP_ADDRESS_LOCAL_PATH = os.path.join(CUSTOM_FILES_PATH,
                                     'ip.json')

##############
# checkpoint #
##############
# best and last checkpoint
BEST_CHECKPOINT = 'best_checkpoint'
LAST_CHECKPOINT = 'last_checkpoint'


# sync files
class SyncFiles(Enum):
    FINISHED = ".finished"
    LOCKFILE = ".lock"
    TRAINER_READY = ".ready"


# s3 postfix
CHECKPOINT_POSTFIX_DIR = "model"
COACH_CHECKPOINT_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                        ".coach_checkpoint")
OLD_COACH_CHECKPOINT_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                            "checkpoint")
DEEPRACER_CHECKPOINT_KEY_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                                "deepracer_checkpoints.json")
FINISHED_FILE_KEY_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                         SyncFiles.FINISHED.value)
LOCKFILE_KEY_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                    SyncFiles.LOCKFILE.value)
TRAINER_READY_KEY_POSTFIX = os.path.join(CHECKPOINT_POSTFIX_DIR,
                                         SyncFiles.TRAINER_READY.value)

# SyncFiles s3 post dict
SYNC_FILES_POSTFIX_DICT = {SyncFiles.FINISHED.value: FINISHED_FILE_KEY_POSTFIX,
                           SyncFiles.LOCKFILE.value: LOCKFILE_KEY_POSTFIX,
                           SyncFiles.TRAINER_READY.value: TRAINER_READY_KEY_POSTFIX}

# {} should be replaced by ./checkpoint_folder/agent_name
CHECKPOINT_LOCAL_DIR_FORMAT = os.path.join("{}")
COACH_CHECKPOINT_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                                  ".coach_checkpoint")
TEMP_COACH_CHECKPOINT_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                                       ".temp_coach_checkpoint")
OLD_COACH_CHECKPOINT_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                                      "checkpoint")
DEEPRACER_CHECKPOINT_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                                      "deepracer_checkpoints.json")
FINISHED_FILE_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                               SyncFiles.FINISHED.value)
LOCKFILE_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                          SyncFiles.LOCKFILE.value)
TRAINER_READY_LOCAL_PATH_FORMAT = os.path.join(CHECKPOINT_LOCAL_DIR_FORMAT,
                                               SyncFiles.TRAINER_READY.value)

# SyncFiles local path dict
SYNC_FILES_LOCAL_PATH_FORMAT_DICT = {SyncFiles.FINISHED.value: FINISHED_FILE_LOCAL_PATH_FORMAT,
                                     SyncFiles.LOCKFILE.value: LOCKFILE_LOCAL_PATH_FORMAT,
                                     SyncFiles.TRAINER_READY.value: TRAINER_READY_LOCAL_PATH_FORMAT}

NUM_MODELS_TO_KEEP = 4
TEMP_RENAME_FOLDER = "./renamed_checkpoint"
# Temporary folder where the model_{}.pb for best_checkpoint_iteration, last_checkpoint_iteration
# and other iterations > last_checkpoint_iteration are stored
SM_MODEL_PB_TEMP_FOLDER = './frozen_models'
