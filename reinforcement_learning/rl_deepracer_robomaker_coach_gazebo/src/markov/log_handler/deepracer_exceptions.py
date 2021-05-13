"""This module should house all custom exceptions"""
import logging

from markov.log_handler.constants import (
    SIMAPP_EVENT_ERROR_CODE_400,
    SIMAPP_EVENT_ERROR_CODE_500,
    SIMAPP_EVENT_SYSTEM_ERROR,
    SIMAPP_EVENT_USER_ERROR,
    SIMAPP_SIMULATION_WORKER_EXCEPTION,
    SIMAPP_TRAINING_WORKER_EXCEPTION,
    SIMAPP_VALIDATION_WORKER_EXCEPTION,
)
from markov.log_handler.exception_handler import log_and_exit
from markov.log_handler.logger import Logger

LOG = Logger(__name__, logging.INFO).get_logger()


class RewardFunctionError(Exception):
    """This exception is for user errors associated with the reward function"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(RewardFunctionError, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the sim app"""
        log_and_exit(
            "Reward function error: {}".format(self.msg),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )


class GenericTrainerException(Exception):
    """This exception is a generic exception for the training worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericTrainerException, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the app"""
        log_and_exit(
            "Training worker failed: {}".format(self.msg),
            SIMAPP_TRAINING_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )


class GenericTrainerError(Exception):
    """This exception is a generic error for the training worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericTrainerError, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the app"""
        log_and_exit(
            "Training worker failed: {}".format(self.msg),
            SIMAPP_TRAINING_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )


class GenericRolloutException(Exception):
    """This exception is a generic exception for the rollout worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericRolloutException, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the sim app"""
        log_and_exit(
            "Rollout worker failed: {}".format(self.msg),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )


class GenericRolloutError(Exception):
    """'This exception is a generic error for the rollout worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericRolloutError, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the sim app"""
        log_and_exit(
            "Rollout worker failed: {}".format(self.msg),
            SIMAPP_SIMULATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )


class GenericValidatorException(Exception):
    """This exception is a generic exception for the validation worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericValidatorException, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the sim app"""
        log_and_exit(
            "Validation worker failed: {}".format(self.msg),
            SIMAPP_VALIDATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_500,
        )


class GenericValidatorError(Exception):
    """'This exception is a generic error for the validation worker"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericValidatorError, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        """Logs the exception to cloud watch and exits the sim app"""
        log_and_exit(
            "Validation worker failed: {}".format(self.msg),
            SIMAPP_VALIDATION_WORKER_EXCEPTION,
            SIMAPP_EVENT_ERROR_CODE_400,
        )


class GenericException(Exception):
    """This exception is a generic exception"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericException, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self, worker):
        """Logs the exception to cloud watch and exits the worker
        worker - String indicating which worker is throwing the exception
        """
        log_and_exit(
            "Validation worker failed: {}".format(self.msg), worker, SIMAPP_EVENT_ERROR_CODE_500
        )


class GenericError(Exception):
    """'This exception is a generic error"""

    def __init__(self, msg):
        """msg - This should be text containing information about what caused
        the exception, for example "cannot divide by zero
        """
        super(GenericError, self).__init__(msg)
        self.msg = msg

    def log_except_and_exit(self, worker):
        """Logs the exception to cloud watch and exits the worker
        worker - String indicating which worker is throwing the exception
        """
        log_and_exit(
            "Validation worker failed: {}".format(self.msg), worker, SIMAPP_EVENT_ERROR_CODE_400
        )


class GenericNonFatalException(Exception):
    """'This exception is a generic non fatal exception that can be resolved by retry.
    In this case, we should log the error and not exit the simapp
    """

    def __init__(
        self,
        error_msg,
        error_code=SIMAPP_EVENT_ERROR_CODE_500,
        error_name=SIMAPP_EVENT_SYSTEM_ERROR,
    ):
        """Initialize the GenericNonFatalException error type.

        Args:
            error_msg (str): The message detailing why we are raising this exception.
            error_code (str, optional): The error code for this exception.
                                        Defaults to SIMAPP_EVENT_ERROR_CODE_500.
            error_name (str, optional): Gives the exception a birth name. Defaults to SIMAPP_EVENT_SYSTEM_ERROR.
        """
        super(GenericNonFatalException, self).__init__()
        self._error_code = error_code
        self._error_name = error_name
        self._error_msg = error_msg

    @property
    def error_code(self):
        """Return a read only version of the error code.

        Returns:
            str: The error code for the current non fatal exception.
        """
        return self._error_code

    @property
    def error_name(self):
        """Return a read only version of the error message.

        Returns:
            str: The error message for the current non fatal exception.
        """
        return self._error_name

    @property
    def error_msg(self):
        """Return a read only version of the error message.

        Returns:
            str: The error message for the current non fatal exception.
        """
        return self._error_msg

    def log_except_and_continue(self):
        """Log the exception and continue.
        TODO: implemement logic here if we decide to log non fatal error metrics
        to cloudwatch.
        """
        error_msg = "DeepRacer non-fatal error - code: {} message: {}".format(
            self._error_code, self._error_msg
        )
        LOG.error(error_msg)
