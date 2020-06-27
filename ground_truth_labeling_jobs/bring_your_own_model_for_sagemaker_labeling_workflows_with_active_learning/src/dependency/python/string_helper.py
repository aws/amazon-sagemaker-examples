import string
import random

from typing import Tuple


def generate_random_string(size=8, chars=string.ascii_letters + string.digits) -> str:
    """
     generate a random string of given characters.
    """
    return ''.join(random.choice(chars) for _ in range(size))


def generate_job_id_and_s3_path(id_prefix, s3_folder_uri,
                                job_type="active-learning") -> Tuple[str, str]:
    """
    generate a pair of job_id and s3_uri where the ouput of the job is to be stored.
        the id_prefix is used as a prefix for the job.
        the s3_folder_uri is the folder within which the output of the job will be stored.
        the job_type can be anything to represent the type of job.
             - "active-learning" job_type is used for training and transform jobs.
             - "labeling-job" job_type is used for manual labeling prefix.
    """
    suffix = generate_random_string()
    job_id = "{}-{}".format(id_prefix,suffix)
    s3_uri = '{}{}-{}/'.format(s3_folder_uri, job_type, suffix)
    return job_id, s3_uri
