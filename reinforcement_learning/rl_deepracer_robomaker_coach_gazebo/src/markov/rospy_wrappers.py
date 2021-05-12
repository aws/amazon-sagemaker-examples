"""This module is intended for any wrappers that are needed for rospy"""
import logging
import time

import rospy
from markov.constants import ROBOMAKER_CANCEL_JOB_WAIT_TIME
from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

ROS_SERVICE_ERROR_MSG_FORMAT = "ROS Service {0} call failed, Re-try count: {1}/{2}: {3}"

logger = Logger(__name__, logging.INFO).get_logger()


class ServiceProxyWrapper(object):
    """This class wraps rospy's ServiceProxy method so that we can wait
    5 minutes if a service throws an exception. This is required to prevent
    our metrics from being flooded since an exception is thrown by service
    calls when the cancel simulation API is called. Because robomaker gives
    us no way of knowing whether or not the exception is real or because the
    sim app is shutting down we have to wait 5 minutes prior logging the exception
    and exiting.
    """

    def __init__(self, service_name, object_type, persistent=False, max_retry_attempts=5):
        """service_name (str): Name of the service to create a client for
        object_type (object): The object type for making a service request
        persistent (bool): flag to whether keep the connection open or not
        max_retry_attempts (int): maximum number of retry
        """
        self.client = rospy.ServiceProxy(service_name, object_type, persistent)
        self._service_name = service_name
        self._max_retry_attempts = max_retry_attempts

    def __call__(self, *argv):
        """Makes a client call for the stored service
        argv (list): Arguments to pass into the client object
        """
        try_count = 0
        while True:
            try:
                return self.client(*argv)
            except TypeError as err:
                log_and_exit(
                    "Invalid arguments for client {}".format(err),
                    SIMAPP_SIMULATION_WORKER_EXCEPTION,
                    SIMAPP_EVENT_ERROR_CODE_500,
                )
            except Exception as ex:
                try_count += 1
                if try_count > self._max_retry_attempts:
                    time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)
                    log_and_exit(
                        "Unable to call service {}".format(ex),
                        SIMAPP_SIMULATION_WORKER_EXCEPTION,
                        SIMAPP_EVENT_ERROR_CODE_500,
                    )
                error_message = ROS_SERVICE_ERROR_MSG_FORMAT.format(
                    self._service_name, str(try_count), str(self._max_retry_attempts), ex
                )
                logger.info(error_message)
