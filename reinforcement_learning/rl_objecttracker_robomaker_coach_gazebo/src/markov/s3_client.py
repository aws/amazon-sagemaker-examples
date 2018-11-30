import boto3
import os
import io
import json
import time

from google.protobuf import text_format
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


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
        print("Uploaded %s model files to S3" % num_files)

    def download_model(self, checkpoint_dir):
        s3_client = self.get_client()
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
                        print("Checkpoint not found. Waiting...")
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
                        print("Downloaded %s model files from S3" % num_files)
                        return True

        except Exception as e:
            raise ValueError("Got exception: %s\n while downloading checkpoint from S3" % e)

    def get_ip(self):
        s3_client = self.get_client()
        self._wait_for_ip_upload()
        try:
            s3_client.download_file(self.bucket, self.config_key, 'ip.json')
            with open("ip.json") as f:
                ip = json.load(f)["IP"]
            return ip
        except Exception as e:
            raise RuntimeError("Cannot fetch IP of redis server running in SageMaker:", e)

    def _wait_for_ip_upload(self, timeout=600):
        s3_client = self.get_client()
        time_elapsed = 0
        while True:
            response = s3_client.list_objects(Bucket=self.bucket, Prefix=self.done_file_key)
            if "Contents" not in response:
                time.sleep(1)
                time_elapsed += 1
                if time_elapsed % 5 == 0:
                    print("Waiting for SageMaker Redis server IP... Time elapsed: %s seconds" % time_elapsed)
                if time_elapsed >= timeout:
                    raise RuntimeError("Cannot retrieve IP of redis server running in SageMaker")
            else:
                return

    def download_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.download_file(self.bucket, s3_key, local_path)
            return True
        except Exception as e:
            return False

    def upload_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.upload_file(Filename=local_path,
                                  Bucket=self.bucket,
                                  Key=s3_key)
            return True
        except Exception as e:
            return False
