'''This takes care of uploading to S3 in a multiprocess environment'''
import os
import re
from multiprocessing import Pool

from markov.s3_client import SageS3Client
from markov.utils import log_and_exit, SIMAPP_SIMULATION_WORKER_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500
from markov.metrics.constants import MULTIPROCESS_S3WRITER_POOL

class S3Writer(object):
    '''This takes care of uploading to S3 in a multiprocess environment'''
    def __init__(self, job_info):
        '''s3_dict - Dictionary containing the required s3 info with keys
                     specified by MetricsS3Keys
        '''
        self.job_info = job_info
        self.upload_num = 0
        self.agent_pattern = r'.*agent/|.*agent_\d+/'

    def _multiprocess_upload_s3(self, s3_bucket, s3_prefix, aws_region, local_file):
        if os.path.exists(local_file):
            s3_client = SageS3Client(bucket=s3_bucket, s3_prefix=s3_prefix, aws_region=aws_region)
            s3_keys = "{}/{}/{}-{}".format(s3_prefix, os.path.dirname(re.sub(self.agent_pattern, '', local_file)),
                                           self.upload_num, os.path.basename(local_file))
            s3_client.upload_file(s3_keys, local_file)

    def upload_to_s3(self):
        ''' This will upload all the files provided parallely using multiprocess
        '''
        # Continue uploading other files if one of the file does not exists
        try:
            multiprocess_pool = Pool(MULTIPROCESS_S3WRITER_POOL)
            multiprocess_pool.starmap(self._multiprocess_upload_s3,
                                      [(job.s3_bucket, job.s3_prefix, job.aws_region, job.local_file)
                                       for job in self.job_info])
            multiprocess_pool.close()
            multiprocess_pool.join()
            _ = [os.remove(job.local_file) for job in self.job_info]
            self.upload_num += 1
        except Exception as ex:
            log_and_exit('Unclassified exception: {}'.format(ex), SIMAPP_SIMULATION_WORKER_EXCEPTION,
                         SIMAPP_EVENT_ERROR_CODE_500)
