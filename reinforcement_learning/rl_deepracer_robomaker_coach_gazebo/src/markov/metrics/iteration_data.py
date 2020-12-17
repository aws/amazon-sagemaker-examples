''' Iteration data that exposes all the S3 Upload job parameter as property'''
class IterationData(object):
    """Iteration data that exposes all the S3 Upload job parameter as property
    """
    def __init__(self, job_name, s3_bucket, s3_prefix, aws_region, local_file):
        """ IterationData params to upload to S3 bucket
        Arguments:
            job_name {[str]} -- [Job name of the file that should be uploaded to s3]
            s3_bucket {[str]} -- [S3 bucket where the job has to be uploaded]
            s3_prefix {[str]} -- [S3 prefix where the job has to be uploaded]
            aws_region {[str]} -- [S3 region where the job has to be uploaded]
            local_file {[str]} -- [Local file that has to be uploaded to s3]
        """
        self._job_name = job_name
        self._s3_bucket = s3_bucket
        self._s3_prefix = s3_prefix
        self._aws_region = aws_region
        self._local_file = local_file

    @property
    def job_name(self):
        """ Job name property
        Returns:
            [str] -- [job name of the s3 writer]
        """
        return self._job_name

    @property
    def s3_bucket(self):
        """ S3 bucket for the job
        Returns:
            [str] -- [S3 bucket where the job has to be uploaded]
        """
        return self._s3_bucket

    @property
    def s3_prefix(self):
        """ S3 prefix for the job
        Returns:
            [str] -- [S3 prefix where the job has to be uploaded]
        """
        return self._s3_prefix

    @property
    def aws_region(self):
        """ S3 region for the job
        Returns:
            [str] -- [S3 region where the job has to be uploaded]
        """
        return self._aws_region

    @property
    def local_file(self):
        """ Local file that has to be uploaded to s3
        Returns:
            [str] -- [Local file that has to be uploaded to s3]
        """
        return self._local_file
