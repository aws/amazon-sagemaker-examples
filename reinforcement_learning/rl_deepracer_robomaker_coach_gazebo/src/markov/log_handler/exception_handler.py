# Handles logging and graceful exit
import datetime
import inspect
import io
import json
import logging
import os
import re
import traceback
from collections import OrderedDict

from markov.log_handler.constants import (
    ERROR_HANDLER_EXCEPTION_FAULT_CODE,
    EXCEPTION_HANDLER_SYNC_FILE,
    FAULT_MAP,
    SIMAPP_ERROR_EXIT,
    SIMAPP_ERROR_HANDLER_EXCEPTION,
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_EVENT_USER_ERROR,
    UNCLASSIFIED_FAULT_CODE,
)
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


def log_and_exit(msg, error_source, error_code):
    """Helper method that logs an exception and exits the application.
    In case of multiple exceptions due to nodes failing, only the first exception will be logged
    using logic to check if the sync file ERROR.txt exists in the environment.

    Args:
        msg (String): The message to be logged
        error_source (String): The source of the error, training worker, rolloutworker, etc
        error_code (String): 4xx or 5xx error

    Returns:
        JSON string data format: Consists of the error log dumped (if sync file not present)
    """
    try:
        s3_crash_status_file_name = os.environ.get("CRASH_STATUS_FILE_NAME", None)
        # TODO: Find an atomic way to check if file is present else create
        # If the sync file is already present, skip logging
        if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
            simapp_exit_gracefully()
        else:
            # Create a sync file ERROR.txt in the environment and log the exception
            with open(EXCEPTION_HANDLER_SYNC_FILE, "w") as sync_file:
                dict_obj = OrderedDict()
                json_format_log = dict()
                fault_code = get_fault_code_for_error(msg)
                file_name = inspect.stack()[1][1].split("/")[-1]
                function_name = inspect.stack()[1][3]
                line_number = inspect.stack()[1][2]
                dict_obj["date"] = str(datetime.datetime.now())
                dict_obj["function"] = "{}::{}::{}".format(file_name, function_name, line_number)
                dict_obj["message"] = msg
                dict_obj["exceptionType"] = error_source
                if error_code == SIMAPP_EVENT_ERROR_CODE_400:
                    dict_obj["eventType"] = SIMAPP_EVENT_USER_ERROR
                else:
                    dict_obj["eventType"] = SIMAPP_EVENT_SYSTEM_ERROR
                dict_obj["errorCode"] = error_code
                # TODO: Include fault_code in the json schema to track faults - pending cloud team assistance
                # dict_obj["faultCode"] = fault_code
                json_format_log["simapp_exception"] = dict_obj
                json_log = json.dumps(json_format_log)
                LOG.error(json_log)
                sync_file.write(json_log)
                # Temporary fault code log
                LOG.error("ERROR: FAULT_CODE: {}".format(fault_code))
            simapp_exit_gracefully(
                json_log=json_log, s3_crash_status_file_name=s3_crash_status_file_name
            )

    except Exception as ex:
        msg = "Exception thrown in logger - log_and_exit: {}".format(ex)
        dict_obj = OrderedDict()
        json_format_log = dict()
        fault_code = ERROR_HANDLER_EXCEPTION_FAULT_CODE
        dict_obj["date"] = str(datetime.datetime.now())
        dict_obj["function"] = "exception_handler.py::log_and_exit::66"
        dict_obj["message"] = msg
        dict_obj["exceptionType"] = SIMAPP_ERROR_HANDLER_EXCEPTION
        dict_obj["eventType"] = SIMAPP_EVENT_SYSTEM_ERROR
        dict_obj["errorCode"] = SIMAPP_EVENT_ERROR_CODE_500
        # TODO: Include fault_code in the json schema to track faults - pending cloud team assistance
        # dict_obj["faultCode"] = fault_code
        json_format_log["simapp_exception"] = dict_obj
        json_log = json.dumps(json_format_log)
        LOG.error(json_log)
        # Temporary fault code log
        LOG.error("ERROR: FAULT_CODE: {}".format(fault_code))
        simapp_exit_gracefully(
            json_log=json_log, s3_crash_status_file_name=s3_crash_status_file_name
        )


def simapp_exit_gracefully(
    simapp_exit=SIMAPP_ERROR_EXIT, json_log=None, s3_crash_status_file_name=None
):
    # simapp exception leading to exiting the system
    # -close the running processes
    # -upload simtrace data to S3
    LOG.info("simapp_exit_gracefully: simapp_exit-{}".format(simapp_exit))
    LOG.info("Terminating simapp simulation...")
    callstack_trace = "".join(traceback.format_stack())
    LOG.info(
        "simapp_exit_gracefully - callstack trace=Traceback (callstack)\n{}".format(callstack_trace)
    )
    exception_trace = traceback.format_exc()
    LOG.info("simapp_exit_gracefully - exception trace={}".format(exception_trace))
    upload_to_s3(json_log=json_log, s3_crash_status_file_name=s3_crash_status_file_name)
    if simapp_exit == SIMAPP_ERROR_EXIT:
        os._exit(1)


# the global variable for upload_to_s3
is_upload_to_s3_called = False


def upload_to_s3(json_log, s3_crash_status_file_name):
    if s3_crash_status_file_name is None or json_log is None:
        LOG.info("simapp_exit_gracefully - skipping s3 upload.")
        return
    # this global variable is added to avoid us running into infinte loop
    # because s3 client could call log and exit as well.
    global is_upload_to_s3_called
    if not is_upload_to_s3_called:
        is_upload_to_s3_called = True
        try:
            # I know this dynamic import can be considered as bad code design
            # however, it's needed to playaround the circular import issue
            # without large scale code change in all places that import log_and_exit
            # TODO: refactor this when we migrate entirely to python 3
            from markov import utils
            from markov.boto.s3.s3_client import S3Client

            LOG.info("simapp_exit_gracefully - first time upload_to_s3 called.")
            s3_client = S3Client()
            s3_key = os.path.normpath(
                os.path.join(os.environ.get("YAML_S3_PREFIX", ""), s3_crash_status_file_name)
            )
            s3_client.upload_fileobj(
                bucket=os.environ.get("YAML_S3_BUCKET", ""),
                s3_key=s3_key,
                fileobj=io.BytesIO(json_log.encode()),
                s3_kms_extra_args=utils.get_s3_extra_args(),
            )
            LOG.info(
                "simapp_exit_gracefully - Successfully uploaded simapp status to \
                      s3 bucket {} with s3 key {}.".format(
                    os.environ.get("YAML_S3_BUCKET", ""), s3_key
                )
            )
        except Exception as ex:
            LOG.error("simapp_exit_gracefully - upload to s3 failed=%s", ex)


def get_fault_code_for_error(msg):
    """Helper method that classifies an error message generated in log_and_exit
    into individual error codes from the maintained FAULT_MAP. If an unseen error
    is seen, it is classified into fault code 0

    Args:
        msg (String): The message to be classified

    Returns:
        String: Consists of classified fault code
    """
    for fault_code, err in FAULT_MAP.items():
        # Match errors stored in FAULT_MAP with the exception thrown
        classified = re.search(r"{}".format(err.lower()), msg.lower()) is not None
        if classified:
            return str(fault_code)
    return UNCLASSIFIED_FAULT_CODE
