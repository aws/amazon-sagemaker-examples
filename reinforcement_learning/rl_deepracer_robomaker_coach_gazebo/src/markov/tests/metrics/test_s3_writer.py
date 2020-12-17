import os
import pytest

# If it fails to import this package it will skip the complete module
# Use this import if its a robomaker unit test
rospy = pytest.importorskip("rospy")

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
