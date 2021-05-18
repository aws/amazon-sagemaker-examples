import os
import pickle

from markov.boto.s3.s3_client import S3Client
from markov.log_handler.deepracer_exceptions import GenericTrainerException
from markov.utils import get_s3_kms_extra_args


class SampleCollector:
    """
    Sample Collector class to collect sample and persist to S3.
    """

    def __init__(
        self,
        bucket,
        s3_prefix,
        region_name,
        max_sample_count=None,
        sampling_frequency=None,
        max_retry_attempts=5,
        backoff_time_sec=1.0,
    ):
        """Sample Collector class to collect sample and persist to S3.

        Args:
            bucket (str): S3 bucket string
            s3_prefix (str): S3 prefix string
            region_name (str): S3 region name
            max_sample_count (int): max sample count
            sampling_frequency (int): sampleing frequency
            max_retry_attempts (int): maximum number of retry attempts for S3 download/upload
            backoff_time_sec (float): backoff second between each retry
        """
        self.max_sample_count = max_sample_count or 0
        self.sampling_frequency = sampling_frequency or 1
        if self.sampling_frequency < 1:
            err_msg = "sampling_frequency must be larger or equal to 1. (Given: {})".format(
                self.sampling_frequency
            )
            raise GenericTrainerException(err_msg)
        self.s3_prefix = s3_prefix

        self._cur_sample_count = 0
        self._cur_frequency = 0
        self._bucket = bucket
        self._s3_client = S3Client(region_name, max_retry_attempts, backoff_time_sec)

    def sample(self, data):
        """Save given data as pickle and upload to s3.
        - collector will stop persisting if the number of samples reached max_sample_count.
        - collector will only persist if sampling_frequency is met.

        Args:
            data (object): The sample data to pickle and upload to S3
        """
        if self._cur_sample_count >= self.max_sample_count:
            return
        self._cur_frequency += 1
        if self._cur_frequency < self.sampling_frequency:
            return

        pickle_filename_format = "sample_{}.pkl"
        pickle_filename = pickle_filename_format.format(self._cur_sample_count)
        try:
            with open(pickle_filename, "wb") as out_f:
                pickle.dump(data, out_f, protocol=2)
        except Exception as ex:
            raise GenericTrainerException("Failed to dump the sample data: {}".format(ex))

        try:
            self._s3_client.upload_file(
                bucket=self._bucket,
                s3_key=os.path.normpath("%s/samples/%s" % (self.s3_prefix, pickle_filename)),
                local_path=pickle_filename,
                s3_kms_extra_args=dict(),
            )
        except Exception as ex:
            raise GenericTrainerException(
                "Failed to upload the sample pickle file to S3: {}".format(ex)
            )
        self._cur_frequency = 0
        self._cur_sample_count += 1
