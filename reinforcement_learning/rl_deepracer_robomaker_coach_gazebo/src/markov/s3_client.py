import boto3
import botocore
import os
import io
import json
import time

from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState

import logging
from markov import utils

logger = utils.Logger(__name__, logging.INFO).get_logger()

class SageS3Client():
    def __init__(self, bucket=None, s3_prefix=None, aws_region=None):
        self.aws_region = aws_region
        self.bucket = bucket
        self.s3_prefix = s3_prefix
        self.config_key = os.path.normpath(s3_prefix + "/ip/ip.json")
        self.hyperparameters_key = os.path.normpath(s3_prefix + "/ip/hyperparameters.json")
        self.done_file_key = os.path.normpath(s3_prefix + "/ip/done")
        self.model_checkpoints_prefix = os.path.normpath(s3_prefix + "/model/") + "/"
        self.lock_file = ".lock"
        logger.info("Initializing SageS3Client...")

    def get_client(self):
        session = boto3.session.Session()
        return session.client('s3', region_name=self.aws_region)

    def _get_s3_key(self, key):
        return os.path.normpath(self.model_checkpoints_prefix + "/" + key)

    def write_ip_config(self, ip):
        s3_client = self.get_client()
        data = {"IP": ip}
        json_blob = json.dumps(data)
        file_handle = io.BytesIO(json_blob.encode())
        file_handle_done = io.BytesIO(b'done')
        s3_client.upload_fileobj(file_handle, self.bucket, self.config_key)
        s3_client.upload_fileobj(file_handle_done, self.bucket, self.done_file_key)

    def upload_hyperparameters(self, hyperparams_json):
        s3_client = self.get_client()
        file_handle = io.BytesIO(hyperparams_json.encode())
        s3_client.upload_fileobj(file_handle, self.bucket, self.hyperparameters_key)

    def upload_model(self, checkpoint_dir):
        s3_client = self.get_client()
        num_files = 0
        for root, dirs, files in os.walk("./" + checkpoint_dir):
            for filename in files:
                abs_name = os.path.abspath(os.path.join(root, filename))
                s3_client.upload_file(abs_name,
                                      self.bucket,
                                      "%s/%s/%s" % (self.s3_prefix, checkpoint_dir, filename))
                num_files += 1

    def download_model(self, checkpoint_dir):
        s3_client = self.get_client()
        filename = "None"
        try:
            filename = os.path.abspath(os.path.join(checkpoint_dir, "checkpoint"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            while True:
                response = s3_client.list_objects_v2(Bucket=self.bucket,
                                                     Prefix=self._get_s3_key(self.lock_file))

                if "Contents" not in response:
                    # If no lock is found, try getting the checkpoint
                    try:
                        s3_client.download_file(Bucket=self.bucket,
                                                Key=self._get_s3_key("checkpoint"),
                                                Filename=filename)
                    except Exception as e:
                        time.sleep(2)
                        continue
                else:
                    time.sleep(2)
                    continue

                ckpt = CheckpointState()
                if os.path.exists(filename):
                    contents = open(filename, 'r').read()
                    text_format.Merge(contents, ckpt)
                    rel_path = ckpt.model_checkpoint_path
                    checkpoint = int(rel_path.split('_Step')[0])

                    response = s3_client.list_objects_v2(Bucket=self.bucket,
                                                         Prefix=self._get_s3_key(rel_path))
                    if "Contents" in response:
                        num_files = 0
                        for obj in response["Contents"]:
                            filename = os.path.abspath(os.path.join(checkpoint_dir,
                                                                    obj["Key"].replace(self.model_checkpoints_prefix,
                                                                                       "")))
                            s3_client.download_file(Bucket=self.bucket,
                                                    Key=obj["Key"],
                                                    Filename=filename)
                            num_files += 1
                        return True

        except Exception as e:
            util.json_format_logger ("{} while downloading the model {} from S3".format(e, filename),
                     **util.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
            return False

    def get_ip(self):
        s3_client = self.get_client()
        self._wait_for_ip_upload()
        try:
            s3_client.download_file(self.bucket, self.config_key, 'ip.json')
            with open("ip.json") as f:
                ip = json.load(f)["IP"]
            return ip
        except Exception as e:
            utils.json_format_logger("Exception [{}] occured, Cannot fetch IP of redis server running in SageMaker. Job failed!".format(e),
                        **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_503))
            sys.exit(1)

    def _wait_for_ip_upload(self, timeout=600):
        s3_client = self.get_client()
        time_elapsed = 0
        while True:
            response = s3_client.list_objects(Bucket=self.bucket, Prefix=self.done_file_key)
            if "Contents" not in response:
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed % 5 == 0:
                    logger.info ("Waiting for SageMaker Redis server IP... Time elapsed: %s seconds" % time_elapsed)
                if time_elapsed >= timeout:
                    #raise RuntimeError("Cannot retrieve IP of redis server running in SageMaker")
                    utils.json_format_logger("Cannot retrieve IP of redis server running in SageMaker. Job failed!",
                        **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_503))
                    sys.exit(1)
            else:
                return

    def download_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.download_file(self.bucket, s3_key, local_path)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info("Exception [{}] occured on download file-{} from s3 bucket-{} key-{}".format(e.response['Error'], local_path, self.bucket, s3_key))
                return False
            else:
                utils.json_format_logger("boto client exception error [{}] occured on download file-{} from s3 bucket-{} key-{}"
                            .format(e.response['Error'], local_path, self.bucket, s3_key),
                            **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_401))
                return False
        except Exception as e:
            utils.json_format_logger("Exception [{}] occcured on download file-{} from s3 bucket-{} key-{}".format(e, local_path, self.bucket, s3_key),
                        **utils.build_user_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_401))
            return False

    def upload_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.upload_file(Filename=local_path,
                                  Bucket=self.bucket,
                                  Key=s3_key)
            return True
        except Exception as e:
            utils.json_format_logger("{} on upload file-{} to s3 bucket-{} key-{}".format(e, local_path, self.bucket, s3_key),
                        **utils.build_system_error_dict(utils.SIMAPP_S3_DATA_STORE_EXCEPTION, utils.SIMAPP_EVENT_ERROR_CODE_500))
            return False
