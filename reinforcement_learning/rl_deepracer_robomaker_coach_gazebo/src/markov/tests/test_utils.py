import pytest
import os
import multiprocessing
import json
import botocore
from markov import utils
from markov.log_handler.constants import (SIMAPP_EVENT_SYSTEM_ERROR,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_ERROR_CODE_500, EXCEPTION_HANDLER_SYNC_FILE)
from markov.s3.s3_client import S3Client
from markov.s3.files.model_metadata import ModelMetadata

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
@pytest.mark.parametrize("error, expected", [("Exception that doesn't contain any keyword", False),
                                             ("Exception that contains keyword checkpoint", True)])
def test_is_user_error(error, expected):
    """This function checks the functionality of is_user_error function
    in markov/utils.py

    <is_user_error> determines whether a value error is caused by an invalid checkpoint or model_metadata
    by looking for keywords 'tensor', 'shape', 'checksum', 'checkpoint', 'model_metadata' in the exception message

    Args:
        error (String): Error message to be parsed
        expected (Boolean): Expected return from function
    """
    assert utils.is_user_error(error) == expected

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
