'''This module is intended for any wrappers that are needed for rospy'''
import time
import rospy
from markov.utils import log_and_exit, ROBOMAKER_CANCEL_JOB_WAIT_TIME, \
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500
class ServiceProxyWrapper(object):
    '''This class wraps rospy's ServiceProxy method so that we can wait
       5 minutes if a service throws an exception. This is required to prevent
       our metrics from being flooded since an exception is thrown by service
       calls when the cancel simulation API is called. Because robomaker gives
       us no way of knowing whether or not the exception is real or because the
       sim app is shutting down we have to wait 5 minutes prior logging the exception
       and exiting.
    '''
    def __init__(self, service_name, object_type):
        '''service_name - Name of the service to create a client for
           object_type - The object type for making a service request
        '''
        self.client = rospy.ServiceProxy(service_name, object_type)

    def __call__(self, *argv):
        ''' Makes a client call for the stored service
            argv - Arguments to pass into the client object
        '''
        try:
            return self.client(*argv)
        except TypeError as err:
            log_and_exit("Invalid arguments for client {}".format(err),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
        except Exception as ex:
            time.sleep(ROBOMAKER_CANCEL_JOB_WAIT_TIME)
            log_and_exit("Unable to call service {}".format(ex),
                         SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500)
