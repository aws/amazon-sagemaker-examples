"""File to store all common constants required by markov"""
from enum import Enum

# SimApp Version
SIMAPP_VERSION_4 = 4.0
SIMAPP_VERSION_3 = 3.0
SIMAPP_VERSION_2 = 2.0
SIMAPP_VERSION_1 = 1.0

# Metrics Version
METRICS_VERSION = 2.0

DEFAULT_COLOR = "Black"
# The robomaker team has asked us to wait 5 minutes to let their workflow cancel
# the simulation job
ROBOMAKER_CANCEL_JOB_WAIT_TIME = 60 * 5
# The number of times to retry a failed boto call
NUM_RETRIES = 5
# The time in seconds till a timeout exception is thrown when attempting to make a connection
# default is 60 seconds
CONNECT_TIMEOUT = 120

BEST_CHECKPOINT = "best_checkpoint"
LAST_CHECKPOINT = "last_checkpoint"

# default park position
DEFAULT_PARK_POSITION = (0.0, 0.0)

# YAML parameters/sagemaker environment parameter for KMS encryption
SAGEMAKER_S3_KMS_CMK_ARN = "s3_kms_cmk_arn"
ROBOMAKER_S3_KMS_CMK_ARN = "S3_KMS_CMK_ARN"
S3_KMS_CMK_ARN_ENV = "S3_KMS_CMK_ARN_ENV"
HYPERPARAMETERS = "hyperparameters"

# Profiler On/Off environment variables
SAGEMAKER_IS_PROFILER_ON = "is_profiler_on"
SAGEMAKER_PROFILER_S3_BUCKET = "profiler_s3_bucket"
SAGEMAKER_PROFILER_S3_PREFIX = "profiler_s3_prefix"
ROBOMAKER_IS_PROFILER_ON = "IS_PROFILER_ON"
ROBOMAKER_PROFILER_S3_BUCKET = "PROFILER_S3_BUCKET"
ROBOMAKER_PROFILER_S3_PREFIX = "PROFILER_S3_PREFIX"


class S3KmsEncryption(Enum):
    """S3 KMS encryption related attributes"""

    SERVER_SIDE_ENCRYPTION = "ServerSideEncryption"
    AWS_KMS = "aws:kms"
    SSE_KMS_KEY_ID = "SSEKMSKeyId"
    ACL = "ACL"
    BUCKET_OWNER_FULL_CONTROL = "bucket-owner-full-control"


class ExplorationTypes(Enum):
    """Exploration type values passed as part of the hyper parameter"""

    CATEGORICAL = "categorical"
    E_GREEDY = "e-greedy"


class LossTypes(Enum):
    """Loss type values passed as part of the hyper parameter"""

    MEAN_SQUARED_ERROR = "mean squared error"
    HUBER = "huber"


class HyperParameterKeys(Enum):
    """This enum contains the keys for the hyper parameters to be
    fed into the agent params.
    """

    BATCH_SIZE = "batch_size"
    NUM_EPOCHS = "num_epochs"
    STACK_SIZE = "stack_size"
    LEARNING_RATE = "lr"
    EXPLORATION_TYPE = "exploration_type"
    E_GREEDY_VALUE = "e_greedy_value"
    EPSILON_STEPS = "epsilon_steps"
    BETA_ENTROPY = "beta_entropy"
    DISCOUNT_FACTOR = "discount_factor"
    LOSS_TYPE = "loss_type"
    NUM_EPISODES_BETWEEN_TRAINING = "num_episodes_between_training"
    TERMINATION_CONDITION_MAX_EPISODES = "term_cond_max_episodes"
    TERMINATION_CONDITION_AVG_SCORE = "term_cond_avg_score"
    SAC_ALPHA = "sac_alpha"


# Profiler paths
TRAINING_WORKER_PROFILER_PATH = "./training_worker_profiler.pstats"
ROLLOUT_WORKER_PROFILER_PATH = "./rollout_worker_profiler.pstats"
