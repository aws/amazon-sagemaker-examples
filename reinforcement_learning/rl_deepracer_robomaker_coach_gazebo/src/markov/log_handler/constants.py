"""This module houses the constants for the log_handler package"""

# Names of the exceptions, errors generated
SIMAPP_SIMULATION_WORKER_EXCEPTION = "simulation_worker.exceptions"
SIMAPP_TRAINING_WORKER_EXCEPTION = "training_worker.exceptions"
SIMAPP_VALIDATION_WORKER_EXCEPTION = "validation_worker.exceptions"
SIMAPP_S3_DATA_STORE_EXCEPTION = "s3_datastore.exceptions"
SIMAPP_SQS_RECEIVE_MESSAGE_EXCEPTION = "sqs_receive_message.exceptions"
SIMAPP_SQS_DELETE_MESSAGE_EXCEPTION = "sqs_delete_message.exceptions"
SIMAPP_ENVIRONMENT_EXCEPTION = "environment.exceptions"
SIMAPP_MEMORY_BACKEND_EXCEPTION = "memory_backend.exceptions"
SIMAPP_SIMULATION_SAVE_TO_MP4_EXCEPTION = "save_to_mp4.exceptions"
SIMAPP_SIMULATION_KINESIS_VIDEO_CAMERA_EXCEPTION = "kinesis_video_camera.exceptions"
SIMAPP_ERROR_HANDLER_EXCEPTION = "error_handler.exceptions"
SIMAPP_CAR_NODE_EXCEPTION = "car_node.exceptions"
SIMAPP_VIRTUAL_EVENT_RACE_EXCEPTION = "virtual_event_race.exceptions"

# Type of errors
SIMAPP_EVENT_SYSTEM_ERROR = "system_error"
SIMAPP_EVENT_USER_ERROR = "user_error"

# Error Code
SIMAPP_EVENT_ERROR_CODE_500 = "500"
SIMAPP_EVENT_ERROR_CODE_400 = "400"

# Constants to keep track of simapp_exit
SIMAPP_DONE_EXIT = 0
SIMAPP_ERROR_EXIT = -1

# Fault code - error map
FAULT_MAP = {
    2: "Unable to call service",
    3: "None type for graph manager",
    4: "Unable to upload finish file",
    5: "Unable to upload checkpoint",
    6: "Unable to download checkpoint",
    7: "Checkpoint not found",
    8: "No checkpoint files",
    9: "Failed to decode the fstring format in reward function",
    10: "Unable to download the reward function code.",
    11: "Reward function code S3 key or bucket not available for S3.",
    12: "Failed to import user's reward_function",
    13: "User modified model",
    14: "Rollout worker value error:",
    15: "Eval worker error: Incorrect arguments passed",
    16: "Eval worker value error",
    17: "Invalid SIM_TRACE data format",
    18: "Write ip config failed to upload",
    19: "Hyperparameters failed to upload",
    20: "Timed out while attempting to retrieve the Redis IP",
    21: "Unable to retrieve redis ip",
    22: "Unable to download file",
    23: "Unable to upload file",
    24: "No rollout data retrieved from the rollout worker",
    25: "NaN detected in loss function, aborting training",
    26: "Received SIGTERM. Checkpointing before exiting",
    27: "An error occured while training",
    28: "Environment Error: Could not retrieve IP address",
    29: "Could not find any frozen model file in the local directory",
    30: "Unable to download best checkpoint",
    31: "Connect timeout on endpoint URL",
    32: "No objects found",
    33: "No checkpoint file found",
    34: "Unable to make model compatible",
    35: "Checkpoint never found",
    36: "Failed to parse model_metadata file",
    37: "Validation worker value error",
    38: "Unable to write metrics to s3: bucket",
    39: "Unable to write metrics to s3, exception",
    40: "Invalid arguments for client",
    41: "Reward function error",
    42: "Training worker failed:",
    43: "Rollout worker failed:",
    44: "Validation worker failed:",
    46: "Exception in Kinesis Video camera ros node:",
    47: "Download params and launch of agent node failed",
    48: "Failed to download model_metadata file: s3_bucket:",
    50: "No VPC attached to instance",
    51: "No Internet connection or ec2 service unavailable",
    52: "Issue with your current VPC stack and IAM roles",
    53: "No Internet connection or s3 service unavailable",
    54: "yaml read error:",
    55: "Exception in the handler function to subscribe to save_mp4 download:",
    56: "Exception in the handler function to unsubscribe from save_mp4 download:",
    57: "Exception in save_mp4 ros node:",
    58: "Iconography image does not exists or corrupt image:",
    59: "S3 writer exception:",
    60: "No checkpoint found:",
    61: "User modified ckpt, unrecoverable dataloss or corruption:",
    62: "ValueError in rename checkpoint",
    63: "Unable to upload profiler data",
    64: "Tracker raised Exception",
    65: "User modified model_metadata.json",
    66: "Exception in downloading file",
    67: "Exception in uploading file",
    68: "Exception in uploading fileobj",
    69: "hyperparameters S3 key or bucket not available for S3.",
    70: "Failed to open and load hyperparameters",
    71: "model_metadata S3 key or bucket not available for S3.",
    72: "yaml file S3 key or bucket not available for S3.",
    73: "Unknown agent type in launch file",
    74: "Ip config S3 prefix or bucket not available for S3.",
    75: "Metrics S3 key or bucket not available for S3.",
    76: "Failed to download hyperparameters file",
    77: "Failed to download ip file",
    78: "Failed to download reward function",
    79: "Exception in listing objects",
    80: "Exception in putting objects",
    81: "Unable to upload fileobj",
    82: "Unable to list objects",
    83: "Unable to put object",
    84: "Exception in uploading .finished file",
    85: "Exception in uploading .lock file",
    86: "Exception in uploading .ready file",
    87: "Unable to delete object from s3",
    88: "Can't download deepracer checkpoint json",
    89: "ready never found",
    90: "Exception in downloading .ready",
    91: "Unable to paginate from s3",
    92: "Unknown action_space_type found while getting action dict",
    93: "Unknown training_algorithm value found while loading action space",
    94: "Unknown training_algorithm found while parsing model_metadata",
    95: "Unknown action_space_type found while parsing model_metadata",
    96: "Unknown exploration_type found in hyper parameters",
    97: "Unsupported minimum and maximum action space bounds for scaling values",
    98: "Action space bounds are incorrect while defining",
    99: "Error while getting action space in",
}

# New error yet to be mapped
UNCLASSIFIED_FAULT_CODE = "0"

# Fault code 1 is dedicated for exception in log_and_exit() in exception_handler.py
ERROR_HANDLER_EXCEPTION_FAULT_CODE = "1"

# Synchronization file used in exception_handler.py
EXCEPTION_HANDLER_SYNC_FILE = "EXCEPTION_HANDLER_SYNC_FILE"
