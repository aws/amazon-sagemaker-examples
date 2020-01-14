'''This module should house all custom exceptions'''
from markov import utils

class RewardFunctionError(Exception):
    '''This exception is for user errors associated with the reward function'''
    def __init__(self, msg):
        '''msg - This should be text containing information about what caused
                 the exception, for example "cannot divide by zero
        '''
        super().__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        '''Logs the exception to cloud watch and exits the sim app'''
        utils.log_and_exit("Reward function error: {}".format(self.msg),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)

class GenericTrainerException(Exception):
    '''This exception is a generic exception for the training worker'''
    def __init__(self, msg):
        '''msg - This should be text containing information about what caused
                 the exception, for example "cannot divide by zero
        '''
        super().__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        '''Logs the exception to cloud watch and exits the app'''
        utils.log_and_exit("Training worker failed: {}".format(self.msg),
                           utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)

class GenericTrainerError(Exception):
    '''This exception is a generic error for the training worker'''
    def __init__(self, msg):
        '''msg - This should be text containing information about what caused
                 the exception, for example "cannot divide by zero
        '''
        super().__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        '''Logs the exception to cloud watch and exits the app'''
        utils.log_and_exit("Training worker failed: {}".format(self.msg),
                           utils.SIMAPP_TRAINING_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)

class GenericRolloutException(Exception):
    '''This exception is a generic exception for the rollout worker'''
    def __init__(self, msg):
        '''msg - This should be text containing information about what caused
                 the exception, for example "cannot divide by zero
        '''
        super().__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        '''Logs the exception to cloud watch and exits the sim app'''
        utils.log_and_exit("Rollout worker failed: {}".format(self.msg),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_500)

class GenericRolloutError(Exception):
    ''''This exception is a generic error for the rollout worker'''
    def __init__(self, msg):
        '''msg - This should be text containing information about what caused
                 the exception, for example "cannot divide by zero
        '''
        super().__init__(msg)
        self.msg = msg

    def log_except_and_exit(self):
        '''Logs the exception to cloud watch and exits the sim app'''
        utils.log_and_exit("Rollout worker failed: {}".format(self.msg),
                           utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                           utils.SIMAPP_EVENT_ERROR_CODE_400)
