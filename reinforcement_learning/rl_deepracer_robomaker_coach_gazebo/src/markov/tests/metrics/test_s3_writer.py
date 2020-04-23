import os
import pytest

# If it fails to import this package it will skip the complete module
# Use this import if its a robomaker unit test
rospy = pytest.importorskip("rospy")

from markov.metrics.s3_writer import S3Writer
from markov.metrics.iteration_data import IterationData

@pytest.fixture
def empty_job_info():
    return list()

@pytest.fixture
def simtrace_job_info(s3_bucket, s3_prefix, aws_region):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    simtrace_path = "{}/test_data/simtrace.csv".format(dir_path)
    with open(simtrace_path, "w") as filepointer:
        filepointer.write("")
    return [IterationData('simtrace', s3_bucket, s3_prefix, aws_region, simtrace_path)]

@pytest.fixture
def video_job_info(s3_bucket, s3_prefix, aws_region):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pip_path = "{}/test_data/pip.mp4".format(dir_path)
    mainview_path = "{}/test_data/45degree.mp4".format(dir_path)
    topview_path = "{}/test_data/simtrace.mp4".format(dir_path)
    for filepath in (pip_path, mainview_path, topview_path):
        with open(filepath, "w") as filepointer:
            filepointer.write("")
    return [IterationData('pip', s3_bucket, s3_prefix, aws_region, pip_path),
            IterationData('45degree', s3_bucket, s3_prefix, aws_region, mainview_path),
            IterationData('topview', s3_bucket, s3_prefix, aws_region, topview_path)]

@pytest.fixture
def empty_s3writer(empty_job_info):
    ''' Empty s3writer for the iteration data '''
    return S3Writer(empty_job_info)

@pytest.fixture
def simtrace_s3writer(simtrace_job_info):
    ''' Empty s3writer for the iteration data '''
    return S3Writer(simtrace_job_info)

@pytest.fixture
def video_s3writer(video_job_info):
    ''' Empty s3writer for the iteration data '''
    return S3Writer(simtrace_job_info)

@pytest.fixture
def s3writer(simtrace_job_info, video_job_info):
    ''' Fixture to instantiate the s3writer class with all jobs'''
    all_jobs = list(simtrace_job_info)
    all_jobs.extend(video_job_info)
    return S3Writer(all_jobs)

@pytest.mark.robomaker
def test_upload_to_s3_empty(empty_s3writer):
    ''' Testing upload_to_s3 with empty job '''
    assert empty_s3writer.upload_num == 0
    assert len(empty_s3writer.job_info) == 0
    assert isinstance(empty_s3writer.job_info, list)
    empty_s3writer.upload_to_s3()
    assert empty_s3writer.upload_num == 1
