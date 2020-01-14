import io
import logging
import os
import json
import time
import boto3
import botocore
from markov import utils

logger = utils.Logger(__name__, logging.INFO).get_logger()

# The amount of time for the sim app to wait for sagemaker to produce
# the ip
SAGEMAKER_WAIT_TIME = 1200 # 20 minutes

class SageS3Client():
    def __init__(self, bucket=None, s3_prefix=None, aws_region=None):
        self.aws_region = aws_region
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.config_key = os.path.normpath(s3_prefix + "/ip/ip.json")
        self.hyperparameters_key = os.path.normpath(s3_prefix + "/ip/hyperparameters.json")
        self.done_file_key = os.path.normpath(s3_prefix + "/ip/done")
        self.model_checkpoints_prefix = os.path.normpath(s3_prefix + "/model/") + "/"
        logger.info("Initializing SageS3Client...")

    def get_client(self):
        session = boto3.session.Session()
        return session.client('s3', region_name=self.aws_region)

    def _get_s3_key(self, key):
        return os.path.normpath(self.model_checkpoints_prefix + "/" + key)

    def write_ip_config(self, ip):
        try:
            s3_client = self.get_client()
            data = {"IP": ip}
            json_blob = json.dumps(data)
            file_handle = io.BytesIO(json_blob.encode())
            file_handle_done = io.BytesIO(b'done')
            s3_client.upload_fileobj(file_handle, self.bucket, self.config_key)
            s3_client.upload_fileobj(file_handle_done, self.bucket, self.done_file_key)
        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Write ip config failed to upload to {}, {}".format(self.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
            utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Write ip config failed to upload to {}, {}".format(self.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()

    def upload_hyperparameters(self, hyperparams_json):
        try:
            s3_client = self.get_client()
            file_handle = io.BytesIO(hyperparams_json.encode())
            s3_client.upload_fileobj(file_handle, self.bucket, self.hyperparameters_key)
        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Hyperparameters failed to upload to {}, {}".format(self.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
            utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Hyperparameters failed to upload to {}, {}".format(self.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()

    def get_ip(self):
        s3_client = self.get_client()
        time_elapsed = 0
        try:
            # Wait for sagemaker to produce the redis ip
            while time_elapsed < SAGEMAKER_WAIT_TIME:
                response = s3_client.list_objects(Bucket=self.bucket, Prefix=self.done_file_key)
                if "Contents" in response:
                    break
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed % 5 == 0:
                    logger.info("Waiting for SageMaker Redis server IP: Time elapsed: %s seconds",
                                time_elapsed)
            if time_elapsed >= SAGEMAKER_WAIT_TIME:
                utils.log_and_exit("Timed out while attempting to retrieve the Redis IP",
                                   utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                   utils.SIMAPP_EVENT_ERROR_CODE_500)
            # Download the ip file
            s3_client.download_file(self.bucket, self.config_key, 'ip.json')
            with open("ip.json") as file:
                ip_file = json.load(file)["IP"]
            return ip_file
        except botocore.exceptions.ClientError as e:
            utils.log_and_exit("Unable to retrieve redis ip from {}: {}"\
                               .format(self.bucket, e.response['Error']['Code']),
                               utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_400)
        except Exception as e:
            utils.log_and_exit("Unable to retrieve redis ip from {}: {}".format(self.bucket, e),
                               utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                               utils.SIMAPP_EVENT_ERROR_CODE_500)

    def download_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.download_file(self.bucket, s3_key, local_path)
            return True
        except botocore.exceptions.ClientError as e:
            # It is possible that the file isn't there in which case we should return fasle and let the client decide the next action
            if e.response['Error']['Code'] == "404":
                return False
            else:
                utils.json_format_logger("Unable to download {} from {}: {}".format(s3_key, self.bucket, e.response['Error']['Code']),
                                         **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                       utils.SIMAPP_EVENT_ERROR_CODE_400))
                utils.simapp_exit_gracefully()
        except Exception as e:
            utils.json_format_logger("Unable to download {} from {}: {}".format(s3_key, self.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
            utils.simapp_exit_gracefully()

    def upload_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.upload_file(Filename=local_path,
                                  Bucket=self.bucket,
                                  Key=s3_key)
            return True
        except botocore.exceptions.ClientError as e:
            utils.json_format_logger("Unable to upload {} to {}: {}".format(s3_key, self.bucket, e.response['Error']['Code']),
                                     **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                   utils.SIMAPP_EVENT_ERROR_CODE_400))
        except Exception as e:
            utils.json_format_logger("Unable to upload {} to {}: {}".format(s3_key, self.bucket, e),
                                     **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION,
                                                                     utils.SIMAPP_EVENT_ERROR_CODE_500))
        return False
