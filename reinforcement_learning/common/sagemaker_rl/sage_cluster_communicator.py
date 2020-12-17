import boto3
import os
import io
import json
import time


class SageClusterCommunicator():
    def __init__(self):
        bucket = os.environ.get("SM_HP_S3_BUCKET", None)
        prefix = os.environ.get("SM_HP_S3_PREFIX", None)
        aws_region = os.environ.get("SM_HP_AWS_REGION", None)
        self.aws_region = boto3.Session().region_name if aws_region is None else aws_region
        if bucket is None or prefix is None:
            bucket, prefix = self._find_s3_output_path()
        self.s3_bucket = bucket
        self.s3_prefix = prefix + "/dist-ray"
        self.ip_key = "MASTER_IP.json"
        self.done_file_key = "CONFIG_DONE"

    def get_client(self):
        session = boto3.session.Session()
        return session.client('s3', region_name=self.aws_region)

    def _get_s3_key(self, key):
        return os.path.normpath(self.s3_prefix + "/config/" + key)

    def _required_environment_param(self, parameter_name):
        SM_TRAINING_ENV = json.loads(os.environ.get("SM_TRAINING_ENV"))
        value = SM_TRAINING_ENV.get(parameter_name, None)
        if not value:
            raise ValueError("Missing enrironment parameter '%s'" % parameter_name)
        return value

    def _find_s3_output_path(self):
        """Looks in SageMaker hyperparameters for the S3 output path.
        Uses SM module directory to extract the output path.
        Returns:
            tuple (bucket, prefix)
        """
        module_dir_s3_path = self._required_environment_param("module_dir")
        if not module_dir_s3_path.startswith('s3://'):
            raise ValueError('Unexpected format for module_dir_s3_path.  Expected "s3://...')
        bucket_prefix = module_dir_s3_path.replace("s3://", "")
        bucket, key = bucket_prefix.split('/', 1)
        prefix = "/".join(key.split("/")[:-2])
        if prefix == "":
            # {bucket}/{job_name}/source/sourcedir.tar.gz structure not present
            prefix = self._required_environment_param("job_name")
        return (bucket, prefix)

    def create_s3_signal(self, signal):
        s3_client = self.get_client()
        s3_client.upload_fileobj(io.BytesIO(b''), self.s3_bucket, self._get_s3_key(signal))

    def wait_for_signals(self, signals, timeout=600, sleep_time=5):
        if len(signals) == 0:
            return
        s3_client = self.get_client()
        time_elapsed = 0
        while True:
            keys_found = 0
            for signal in signals:
                response = s3_client.list_objects(Bucket=self.s3_bucket, Prefix=self._get_s3_key(signal))
                if "Contents" in response:
                    keys_found += 1
            if keys_found != len(signals):
                time.sleep(sleep_time)
                time_elapsed += sleep_time
                if time_elapsed >= timeout:
                    raise RuntimeError(
                        "Could not find all the signals: %s for last %s seconds" % (signals, time_elapsed))
            else:
                print("Received all signal[s]: %s" % signals)
                return

    def write_host_config(self, ip, host_name):
        s3_client = self.get_client()
        data = {"IP": ip, "HOST_NAME": host_name}
        json_blob = json.dumps(data)
        file_handle = io.BytesIO(json_blob.encode())
        file_handle_done = io.BytesIO(b'done')
        s3_client.upload_fileobj(file_handle, self.s3_bucket, self._get_s3_key(self.ip_key))
        s3_client.upload_fileobj(file_handle_done, self.s3_bucket, self._get_s3_key(self.done_file_key))

    def get_master_config(self):
        s3_client = self.get_client()
        self._wait_for_ip_upload()
        try:
            s3_client.download_file(self.s3_bucket, self._get_s3_key(self.ip_key), 'ip.json')
            with open("ip.json") as f:
                json_obj = json.load(f)
                ip = json_obj["IP"]
                host_name = json_obj["HOST_NAME"]
            return ip, host_name
        except Exception as e:
            raise RuntimeError("Cannot fetch IP of redis server running in SageMaker:", e)

    def _wait_for_ip_upload(self, timeout=600):
        s3_client = self.get_client()
        time_elapsed = 0
        while True:
            response = s3_client.list_objects(Bucket=self.s3_bucket, Prefix=self._get_s3_key(self.done_file_key))
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
            s3_client.download_file(self.s3_bucket, s3_key, local_path)
            return True
        except Exception as e:
            return False

    def upload_file(self, s3_key, local_path):
        s3_client = self.get_client()
        try:
            s3_client.upload_file(Filename=local_path,
                                  Bucket=self.s3_bucket,
                                  Key=s3_key)
            return True
        except Exception as e:
            return False
