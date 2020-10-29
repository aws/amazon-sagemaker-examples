""" The script takes care of testing the functionality of exception_handler.py
"""
import pytest
import os
import multiprocessing
import json

from markov.log_handler import exception_handler
from markov.log_handler.constants import (SIMAPP_TRAINING_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_SYSTEM_ERROR, SIMAPP_S3_DATA_STORE_EXCEPTION,
                                          SIMAPP_SIMULATION_WORKER_EXCEPTION,
                                          SIMAPP_EVENT_USER_ERROR, SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_EVENT_ERROR_CODE_400, EXCEPTION_HANDLER_SYNC_FILE)

@pytest.mark.robomaker
@pytest.mark.parametrize("message, exceptionType, eventType, errorCode",
                         [("Sample Simulation Worker Exception", SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           SIMAPP_EVENT_SYSTEM_ERROR, SIMAPP_EVENT_ERROR_CODE_500),
                          ("Sample Simulation Worker Exception", SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           SIMAPP_EVENT_USER_ERROR, SIMAPP_EVENT_ERROR_CODE_400)])
def test_log_and_exit_robomaker(message, exceptionType, eventType, errorCode):
    """The test function checks if the log_and_exit() function from exception_handler.py
    once called inside robomaker environment, is able to log the appropriate error message and
    abort the entire program.

    The log_and_exit is tested using another process since we abort the program
    with os exit when we call the function with exit code 1.
    The exception stored in the sync file "EXCEPTION_HANDLER_SYNC_FILE"
    is parsed to check whether the appropriate message is logged.
    The sync file is useful to do this because when we run log_and_exit in multiprocess,
    once the program aborts, all information along with the error logged on stderr is lost.

    Args:
      message: Error message that is to be logged
      exceptionType: The exception type
      eventType: Whether its a system or user error (test if this is decided properly)
      errorCode: Error code (400 or 500)
    """
    # Remove any sync file generated because of other tests generating exceptions
    if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    proc = multiprocessing.Process(target=exception_handler.log_and_exit,
                                   args=(message, exceptionType, errorCode))
    proc.start()
    proc.join()
    assert os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE)
    try:
        with open(EXCEPTION_HANDLER_SYNC_FILE, 'r') as sync_file:
            captured_log = json.loads(sync_file.read())
    finally:
        # Remove the sync file for next test
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    assert not proc.is_alive()
    assert proc.exitcode == 1
    assert captured_log['simapp_exception']['message'] == message
    assert captured_log['simapp_exception']['exceptionType'] == exceptionType
    assert captured_log['simapp_exception']['eventType'] == eventType
    assert captured_log['simapp_exception']['errorCode'] == errorCode

@pytest.mark.sagemaker
@pytest.mark.parametrize("message, exceptionType, eventType, errorCode",
                         [("Sample Training Worker Exception", SIMAPP_TRAINING_WORKER_EXCEPTION,
                           SIMAPP_EVENT_SYSTEM_ERROR, SIMAPP_EVENT_ERROR_CODE_500),
                          ("Sample Training Worker Exception", SIMAPP_TRAINING_WORKER_EXCEPTION,
                           SIMAPP_EVENT_USER_ERROR, SIMAPP_EVENT_ERROR_CODE_400)])
def test_log_and_exit_sagemaker(message, exceptionType, eventType, errorCode):
    """The test function checks if the log_and_exit() function from exception_handler.py
    once called inside sagemaker environment, is able to log the appropriate error message and
    abort the entire program.

    The log_and_exit is tested using another process since we abort the program
    with os exit when we call the function with exit code 1.
    The exception stored in the sync file "EXCEPTION_HANDLER_SYNC_FILE"
    is parsed to check whether the appropriate message is logged.
    The sync file is useful to do this because when we run log_and_exit in multiprocess,
    once the program aborts, all information along with the error logged on stderr is lost.

    Args:
      message: Error message that is to be logged
      exceptionType: The exception type
      eventType: Whether its a system or user error (test if this is decided properly)
      errorCode: Error code (400 or 500)
    """
    # Remove any sync file generated because of other tests generating exceptions
    if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    proc = multiprocessing.Process(target=exception_handler.log_and_exit,
                                   args=(message, exceptionType, errorCode))
    proc.start()
    proc.join()
    assert os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE)
    try:
        with open(EXCEPTION_HANDLER_SYNC_FILE, 'r') as sync_file:
            captured_log = json.loads(sync_file.read())
    finally:
        # Remove the sync file for next test
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    assert not proc.is_alive()
    assert proc.exitcode == 1
    assert captured_log['simapp_exception']['message'] == message
    assert captured_log['simapp_exception']['exceptionType'] == exceptionType
    assert captured_log['simapp_exception']['eventType'] == eventType
    assert captured_log['simapp_exception']['errorCode'] == errorCode

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_log_and_exit_multiple():
    """The test function checks if multiple exceptions are thrown, only the first exception
    thrown should get logged.
    """
    # Remove any sync file generated because of other tests generating exceptions
    if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    message1 = "Sample DataStore Exception 1"
    exceptionType1 = SIMAPP_S3_DATA_STORE_EXCEPTION
    eventType1 = SIMAPP_EVENT_SYSTEM_ERROR
    errorCode1 = SIMAPP_EVENT_ERROR_CODE_500
    # Throwing the first exception and logging it
    proc1 = multiprocessing.Process(target=exception_handler.log_and_exit,
                                    args=(message1, exceptionType1, errorCode1))
    proc1.start()
    proc1.join()
    assert os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE)
    with open(EXCEPTION_HANDLER_SYNC_FILE, 'r') as sync_file:
        captured_log = json.loads(sync_file.read())
    assert not proc1.is_alive()
    assert proc1.exitcode == 1
    assert captured_log['simapp_exception']['message'] == message1
    assert captured_log['simapp_exception']['exceptionType'] == exceptionType1
    assert captured_log['simapp_exception']['eventType'] == eventType1
    assert captured_log['simapp_exception']['errorCode'] == errorCode1
    # Throwing the second exception without removing the sync file
    # The error shouldn't be logged, instead direct SIMAPP exit
    message2 = "Sample DataStore Exception 2"
    exceptionType2 = SIMAPP_S3_DATA_STORE_EXCEPTION
    eventType2 = SIMAPP_EVENT_SYSTEM_ERROR
    errorCode2 = SIMAPP_EVENT_ERROR_CODE_400
    proc2 = multiprocessing.Process(target=exception_handler.log_and_exit,
                                    args=(message2, exceptionType2, errorCode2))
    proc2.start()
    proc2.join()
    assert os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE)
    try:
        with open(EXCEPTION_HANDLER_SYNC_FILE, 'r') as sync_file:
            captured_log = json.loads(sync_file.read())
    finally:
        # Remove the sync file
        os.remove(EXCEPTION_HANDLER_SYNC_FILE)
    assert not proc2.is_alive()
    assert proc2.exitcode == 1
    assert captured_log['simapp_exception']['message'] == message1
    assert captured_log['simapp_exception']['exceptionType'] == exceptionType1
    assert captured_log['simapp_exception']['eventType'] == eventType1
    assert captured_log['simapp_exception']['errorCode'] == errorCode1

@pytest.mark.robomaker
@pytest.mark.sagemaker
@pytest.mark.parametrize("message, fault_code",
                         [("User modified ckpt, unrecoverable dataloss or corruption:", "61"),
                          ("Unseen error while testing", "0")])
def test_get_fault_code_for_error(message, fault_code):
    """The test function checks if get_fault_code_for_error() in exception_handler.py appropriately
    matches the error message with the fault_code from FAULT_MAP.

    In case of an unmapped exception, it should provide the fault_code 0

    Args:
      message: The error message generated
      fault_code: Corresponding fault_code
    """
    assert exception_handler.get_fault_code_for_error(message) == fault_code


@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_simapp_exit_gracefully():
    """This function tests if the simapp_exit_gracefully() function in exception_handler.py
    exits and aborts the program.

    The simapp_exit_gracefully is tested using another process since we abort the program
    with os exit when we call the function with exit code 1.
    """
    proc = multiprocessing.Process(target=exception_handler.simapp_exit_gracefully)
    proc.start()
    proc.join()
    assert not proc.is_alive()
    assert proc.exitcode == 1
