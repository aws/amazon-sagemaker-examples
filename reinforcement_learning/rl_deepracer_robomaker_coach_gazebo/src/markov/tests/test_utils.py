import pytest
import os
import multiprocessing
import json
import botocore
from markov import utils
from markov.s3_client import SageS3Client
from markov.log_handler.constants import (SIMAPP_EVENT_SYSTEM_ERROR,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500, EXCEPTION_HANDLER_SYNC_FILE)

@pytest.mark.robomaker
def test_test_internet_connection(aws_region):
    """This function checks the functionality of test_internet_connection
    function in markov/utils.py

    If an exception is generated, log_and_exit will be called within the
    function and the test will fail.

    Args:
        aws_region (String): AWS_REGION passed from fixture
    """
    utils.test_internet_connection(aws_region)

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_load_model_metadata(s3_bucket, s3_prefix, aws_region, model_metadata_s3_key):
    """This function checks the functionality of load_model_metadata function
    in markov/utils.py

    The function checks if model_metadata.json file is downloaded into the required directory.
    If the function fails, it will generate an exception which will call log_and_exit internally.
    Hence the test will fail.

    Args:
        s3_bucket (String): S3_BUCKET
        s3_prefix (String): S3_PREFIX
        aws_region (String): AWS_REGION
        model_metadata_s3_key (String): MODEL_METADATA_S3_KEY
    """
    s3_client = SageS3Client(bucket=s3_bucket, s3_prefix=s3_prefix, aws_region=aws_region)
    model_metadata_local_path = 'test_model_metadata.json'
    utils.load_model_metadata(s3_client, model_metadata_s3_key, model_metadata_local_path)
    assert os.path.isfile(model_metadata_local_path)
    # Remove file downloaded
    if os.path.isfile(model_metadata_local_path):
        os.remove(model_metadata_local_path)

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_has_current_ckpnt_name(s3_bucket, s3_prefix, aws_region):
    """This function checks the functionality of has_current_ckpnt_name function
    in markov/utils.py

    <utils.has_current_ckpnt_name> checks if the checkpoint key (.coach_checkpoint) is present in S3

    Args:
        s3_bucket (String): S3_BUCKET
        s3_prefix (String): S3_PREFIX
        aws_region (String): AWS_REGION
    """
    assert utils.has_current_ckpnt_name(s3_bucket, s3_prefix, aws_region)

@pytest.mark.robomaker
@pytest.mark.sagemaker
@pytest.mark.parametrize("error, expected", [("Exception that doesn't contain any keyword", False),
                                             ("Exception that contains keyword checkpoint", True)])
def test_is_error_bad_ckpnt(error, expected):
    """This function checks the functionality of is_error_bad_ckpnt function
    in markov/utils.py

    <is_error_bad_ckpnt> determines whether a value error is caused by an invalid checkpoint
    by looking for keywords 'tensor', 'shape', 'checksum', 'checkpoint' in the exception message

    Args:
        error (String): Error message to be parsed
        expected (Boolean): Expected return from function
    """
    assert utils.is_error_bad_ckpnt(error) == expected

@pytest.mark.robomaker
@pytest.mark.parametrize("racecar_num, racer_names", [(1, ['racecar']),
                                                      (2, ['racecar_0', 'racecar_1'])])
def test_get_racecar_names(racecar_num, racer_names):
    """This function checks the functionality of get_racecar_names function
    in markov/utils.py

    Args:
        racecar_num (int): The number of racecars
        racer_names (List): Returned list of racecar names
    """
    assert utils.get_racecar_names(racecar_num) == racer_names

@pytest.mark.robomaker
@pytest.mark.parametrize("racecar_name, racecar_num", [('racecar', None),
                                                       ('racecar_1', 1)])
def test_get_racecar_idx(racecar_name, racecar_num):
    """This function checks the functionality of get_racecar_idx function
    in markov/utils.py

    Args:
        racecar_name (String): The name of racecar
        racecar_num: If single car race, returns None else returns the racecar number
    """
    assert utils.get_racecar_idx(racecar_name) == racecar_num

@pytest.mark.robomaker
def test_get_racecar_idx_exception():
    """This function checks the functionality of get_racecar_idx function
    in markov/utils.py when exception is generated if wrong format passed
    """
    # Remove any sync file generated because of other tests generating exceptions
    if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    err_message = "racecar name should be in format racecar_x. However, get"
    proc = multiprocessing.Process(target=utils.get_racecar_idx,
                                   args=('1_racecar', ))
    proc.start()
    proc.join()
    assert os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE)
    try:
        with open(EXCEPTION_HANDLER_SYNC_FILE, 'r') as sync_file:
            captured_log = json.loads(sync_file.read())
    finally:
        # Remove the sync file created due to log_and_exit
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    assert not proc.is_alive()
    assert proc.exitcode == 1
    assert err_message in captured_log['simapp_exception']['message']
    assert captured_log['simapp_exception']['exceptionType'] == SIMAPP_SIMULATION_WORKER_EXCEPTION
    assert captured_log['simapp_exception']['eventType'] == SIMAPP_EVENT_SYSTEM_ERROR
    assert captured_log['simapp_exception']['errorCode'] == SIMAPP_EVENT_ERROR_CODE_500

@pytest.mark.robomaker
def test_force_list(s3_bucket):
    """This function checks the functionality of force_list function
    in markov/utils.py

    Args:
        s3_bucket (String): S3_BUCKET
    """
    assert utils.force_list(s3_bucket) == [s3_bucket]

@pytest.mark.robomaker
def test_get_boto_config():
    """This function checks the functionality of get_boto_config function
    in markov/utils.py
    """
    utils.get_boto_config()
