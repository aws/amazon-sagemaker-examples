'''File to store all constants required by markov/utils.py'''
from enum import Enum

# SimApp Version
SIMAPP_VERSION = "2.0"
DEFAULT_COLOR = "Black"
# The robomaker team has asked us to wait 5 minutes to let their workflow cancel
# the simulation job
ROBOMAKER_CANCEL_JOB_WAIT_TIME = 60 * 5
# The current checkpoint key
CHKPNT_KEY_SUFFIX = "model/.coach_checkpoint"
# This is the key for the best checkpoint
DEEPRACER_CHKPNT_KEY_SUFFIX = "model/deepracer_checkpoints.json"
# The number of times to retry a failed boto call
NUM_RETRIES = 5

BEST_CHECKPOINT = 'best_checkpoint'
LAST_CHECKPOINT = 'last_checkpoint'

# YAML parameters/sagemaker environment parameter for KMS encryption
SAGEMAKER_S3_KMS_CMK_ARN = "s3_kms_cmk_arn"
ROBOMAKER_S3_KMS_CMK_ARN = "S3_KMS_CMK_ARN"
S3_KMS_CMK_ARN_ENV = "S3_KMS_CMK_ARN_ENV"
HYPERPARAMETERS = "hyperparameters"

class S3KmsEncryption(Enum):
    """ S3 KMS encryption related attributes
    """
    SERVER_SIDE_ENCRYPTION = "ServerSideEncryption"
    AWS_KMS = "aws:kms"
    SSE_KMS_KEY_ID = "SSEKMSKeyId"
    ACL = "ACL"
    BUCKET_OWNER_FULL_CONTROL = "bucket-owner-full-control"
