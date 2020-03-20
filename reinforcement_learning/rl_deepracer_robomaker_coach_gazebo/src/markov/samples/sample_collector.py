import pickle
import os
from markov.deepracer_exceptions import GenericTrainerException

class SampleCollector:
    """
    Sample Collector class to collect sample and persist to S3.
    """
    def __init__(self, s3_client, s3_prefix, max_sample_count=None, sampling_frequency=None):
        self.max_sample_count = max_sample_count or 0
        self.sampling_frequency = sampling_frequency or 1
        if self.sampling_frequency < 1:
            err_msg = "sampling_frequency must be larger or equal to 1. (Given: {})".format(self.sampling_frequency)
            raise GenericTrainerException(err_msg)
        self.s3_client = s3_client
        self.s3_prefix = s3_prefix

        self._cur_sample_count = 0
        self._cur_frequency = 0

    """
    Save given data as pickle and upload to s3.
    - collector will stop persisting if the number of samples reached max_sample_count.
    - collector will only persist if sampling_frequency is met.
    
    Args:
        data (object): The sample data to pickle and upload to S3
    """
    def sample(self, data):
        if self._cur_sample_count >= self.max_sample_count:
            return
        self._cur_frequency += 1
        if self._cur_frequency < self.sampling_frequency:
            return

        pickle_filename_format = 'sample_{}.pkl'
        pickle_filename = pickle_filename_format.format(self._cur_sample_count)
        try:
            with open(pickle_filename, 'wb') as out_f:
                pickle.dump(data, out_f, protocol=2)
        except Exception as ex:
            raise GenericTrainerException('Failed to dump the sample data: {}'.format(ex))

        try:
            self.s3_client.upload_file(os.path.normpath("%s/samples/%s" % (self.s3_prefix, pickle_filename)),
                                       pickle_filename)
        except Exception as ex:
            raise GenericTrainerException('Failed to upload the sample pickle file to S3: {}'.format(ex))
        self._cur_frequency = 0
        self._cur_sample_count += 1

