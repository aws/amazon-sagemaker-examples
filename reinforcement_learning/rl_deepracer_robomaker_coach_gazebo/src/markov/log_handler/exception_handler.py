# Handles logging and graceful exit
import json
import logging
import os
import datetime
import inspect
from collections import OrderedDict
import traceback
import re
from markov.log_handler.constants import (SIMAPP_ERROR_HANDLER_EXCEPTION, SIMAPP_EVENT_SYSTEM_ERROR,
                                          SIMAPP_EVENT_USER_ERROR, SIMAPP_EVENT_ERROR_CODE_500,
                                          SIMAPP_EVENT_ERROR_CODE_400, SIMAPP_ERROR_EXIT, FAULT_MAP,
                                          UNCLASSIFIED_FAULT_CODE, EXCEPTION_HANDLER_SYNC_FILE,
                                          ERROR_HANDLER_EXCEPTION_FAULT_CODE)
from markov.constants import SIMAPP_VERSION
from markov.log_handler.logger import Logger

logger = Logger(__name__, logging.INFO).get_logger()

def log_and_exit(msg, error_source, error_code):
    '''Helper method that logs an exception and exits the application.
    In case of multiple exceptions due to nodes failing, only the first exception will be logged
    using logic to check if the sync file ERROR.txt exists in the environment.

    Args:
        msg (String): The message to be logged
        error_source (String): The source of the error, training worker, rolloutworker, etc
        error_code (String): 4xx or 5xx error

    Returns:
        JSON string data format: Consists of the error log dumped (if sync file not present)
    '''
    try:
        #TODO: Find an atomic way to check if file is present else create
        # If the sync file is already present, skip logging
        if os.path.isfile(EXCEPTION_HANDLER_SYNC_FILE):
            simapp_exit_gracefully()
        else:
            # Create a sync file ERROR.txt in the environment and log the exception
            with open(EXCEPTION_HANDLER_SYNC_FILE, 'w') as sync_file:
                dict_obj = OrderedDict()
                json_format_log = dict()
                fault_code = get_fault_code_for_error(msg)
                file_name = inspect.stack()[1][1].split("/")[-1]
                function_name = inspect.stack()[1][3]
                line_number = inspect.stack()[1][2]
                dict_obj['version'] = SIMAPP_VERSION
                dict_obj['date'] = str(datetime.datetime.now())
                dict_obj['function'] = "{}::{}::{}".format(file_name, function_name, line_number)
                dict_obj['message'] = msg
                dict_obj["exceptionType"] = error_source
                if error_code == SIMAPP_EVENT_ERROR_CODE_400:
                    dict_obj["eventType"] = SIMAPP_EVENT_USER_ERROR
                else:
                    dict_obj["eventType"] = SIMAPP_EVENT_SYSTEM_ERROR
                dict_obj["errorCode"] = error_code
                #TODO: Include fault_code in the json schema to track faults - pending cloud team assistance
                #dict_obj["faultCode"] = fault_code
                json_format_log["simapp_exception"] = dict_obj
                logger.error(json.dumps(json_format_log))
                # Temporary fault code log
                logger.error("ERROR: FAULT_CODE: {}".format(fault_code))
                sync_file.write(json.dumps(json_format_log))
            simapp_exit_gracefully()

    except Exception as ex:
        msg = "Exception thrown in logger - log_and_exit: {}".format(ex)
        dict_obj = OrderedDict()
        json_format_log = dict()
        fault_code = ERROR_HANDLER_EXCEPTION_FAULT_CODE
        dict_obj['version'] = SIMAPP_VERSION
        dict_obj['date'] = str(datetime.datetime.now())
        dict_obj['function'] = 'exception_handler.py::log_and_exit::66'
        dict_obj['message'] = msg
        dict_obj["exceptionType"] = SIMAPP_ERROR_HANDLER_EXCEPTION
        dict_obj["eventType"] = SIMAPP_EVENT_SYSTEM_ERROR
        dict_obj["errorCode"] = SIMAPP_EVENT_ERROR_CODE_500
        #TODO: Include fault_code in the json schema to track faults - pending cloud team assistance
        #dict_obj["faultCode"] = fault_code
        json_format_log["simapp_exception"] = dict_obj
        logger.error(json.dumps(json_format_log))
        # Temporary fault code log
        logger.error("ERROR: FAULT_CODE: {}".format(fault_code))
        simapp_exit_gracefully()

def simapp_exit_gracefully(simapp_exit=SIMAPP_ERROR_EXIT):
    #simapp exception leading to exiting the system
    # -close the running processes
    # -upload simtrace data to S3
    logger.info("simapp_exit_gracefully: simapp_exit-{}".format(simapp_exit))
    logger.info("Terminating simapp simulation...")
    callstack_trace = ''.join(traceback.format_stack())
    logger.info("simapp_exit_gracefully - callstack trace=Traceback (callstack)\n{}".format(callstack_trace))
    exception_trace = traceback.format_exc()
    logger.info("simapp_exit_gracefully - exception trace={}".format(exception_trace))

    if simapp_exit == SIMAPP_ERROR_EXIT:
        os._exit(1)

def get_fault_code_for_error(msg):
    '''Helper method that classifies an error message generated in log_and_exit 
    into individual error codes from the maintained FAULT_MAP. If an unseen error
    is seen, it is classified into fault code 0

    Args:
        msg (String): The message to be classified

    Returns:
        String: Consists of classified fault code
    '''
    for fault_code, err in FAULT_MAP.items():
        # Match errors stored in FAULT_MAP with the exception thrown
        classified = re.search(r"{}".format(err.lower()), msg.lower()) is not None
        if classified:
            return str(fault_code)
    return UNCLASSIFIED_FAULT_CODE
