import json
import logging
import boto3
from pathlib import Path
import shutil
import os
import pandas as pd

logger = logging.getLogger(__name__)


def validate_experience(experience):
    """
    Validate the collected experience has required keys.
    """
    keys = ["shared_context", "actions_context" ,"action_prob", "action", "reward", "user_id"]
    for key in keys:
        is_valid = key in experience
        if not is_valid:
            return False
    return True


class CSVReader():
    """Reader object that loads experiences from CSV file chunks.
    The input files will be read from in an random order."""

    def __init__(self, input_files):
        self.files = input_files

    def get_iterator(self):
        for file in self.files:
            reader = pd.read_csv(file, chunksize=1000)
            for df in reader:
                df_no_nans = df.dropna()
                for line in df_no_nans.iterrows():
                    line_dict = line[1].to_dict()
                    yield line_dict


class JsonLinesReader():
    """Reader object that loads experiences from JSON file chunks.
    The input files will be read from in an random order."""

    def __init__(self, input_files):
        self.files = input_files
        self.cur_file = None
        self.cur_index = 0
        self.max_index = len(input_files) - 1
        self.done = False

    def get_experience(self):
        line = self._next_line()
        experience = self._try_parse(line)
        while not experience and not self.done:
            logger.debug("Skipping empty line in {}".format(self.cur_file))
            experience = self._try_parse(self._next_line())
        return experience

    def _try_parse(self, line):
        if line is None or line.strip() == '':
            return None
        try:
            line_json = json.loads(line.strip())
            assert "observation" in line_json, "observation not found in record"
            assert "action" in line_json, "action not found in record"
            assert "reward" in line_json, "reward not found in record"
            assert "prob" in line_json, "prob not found in record"
            return line_json
        except Exception:
            logger.exception("Ignoring corrupt json record in {}: {}".format(
                self.cur_file, line))
            return None

    def _next_line(self):
        if not self.cur_file:
            self.cur_file = self._next_file()
            if self.done is True:
                return None
        line = self.cur_file.readline()
        tries = 0
        while not line and tries < 100:
            tries += 1
            self.cur_file.close()
            self.cur_file = self._next_file()
            if self.done is True:
                return None
            line = self.cur_file.readline()
            if not line:
                logger.debug("Ignoring empty file {}".format(self.cur_file))
        if not line:
            raise ValueError("Failed to read next line from files: {}".format(
                self.files))
        return line

    def _next_file(self):
        if self.cur_index > self.max_index:
            self.done = True
            return None
        path = self.files[self.cur_index]
        self.cur_index += 1
        return open(path, "r")


def get_vw_model(disk_path=None):
    """
    Returns a tuple (str, str) of metadata string and model weights URL on disk.
    """
    sagemaker_model_path = Path(disk_path)
    meta_files = list(sagemaker_model_path.rglob("vw.metadata"))
    if len(meta_files) == 0:
        raise ValueError("Algorithm Error: 'vw.metadata' not found in model files.")
    metadata_path = meta_files[0]

    model_files = list(sagemaker_model_path.rglob("vw.model"))
    if len(model_files) == 0:
        raise ValueError("Algorithm Error: 'vw.model' not found in model files.")
    model_path = model_files[0]
    return metadata_path.as_posix(), model_path.as_posix()


def extract_model(tar_gz_folder):
    """
    This function extracts the model.tar.gz and then
    returns a tuple (str, str) of metadata string and model weights URL on disk.
    """
    shutil.unpack_archive(filename=os.path.join(tar_gz_folder, "model.tar.gz"), extract_dir=tar_gz_folder)
    return get_vw_model(tar_gz_folder)


def parse_s3_uri(uri):
    uri = uri.replace("s3://", "")
    bucket, *key = uri.split("/")
    file_name = key[-1]
    key = "/".join(key)
    return bucket, key, file_name


def download_manifest_data(manifest_file_path, output_dir):
    """
    Download the s3 files contained in a manifest file.
    """
    with open(manifest_file_path.as_posix()) as f:
        manifest = json.load(f)
    s3_prefix = manifest[0]["prefix"]
    s3 = boto3.client('s3')
    for file in manifest[1:]:
        s3_uri = os.path.join(s3_prefix, file)
        bucket, key, file_name = parse_s3_uri(s3_uri)
        output_file = os.path.join(output_dir.as_posix(), file_name)
        s3.download_file(bucket, key, output_file)
        print("Downloaded file ", output_file)
