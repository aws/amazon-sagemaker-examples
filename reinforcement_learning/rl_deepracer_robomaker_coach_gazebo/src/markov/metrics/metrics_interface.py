'''This module defines the interface for metrics, this is the data we typically upload to
    s3
'''
import abc

class MetricsInterface(object, metaclass=abc.ABCMeta):
    def upload_episode_metrics(self):
        '''Uploads the desired episode metrics to s3
           metrics - Dictionary of metrics to upload
        '''
        raise NotImplementedError('Metrics class must be able to upload episode metrics')

    def upload_step_metrics(self, metrics):
        '''Uploads step metrics to s3
           metrics - Dictionary of metrics to upload
        '''
        raise NotImplementedError('Metrics class must be able to upload step metrics')

    def reset(self):
        '''Reset the desired class data'''
        raise NotImplementedError('Metrics class must be able to reset')
