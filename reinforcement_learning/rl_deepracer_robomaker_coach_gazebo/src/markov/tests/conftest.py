import pytest
from markov.tests import test_constant

@pytest.fixture
def s3_bucket():
    return test_constant.S3_BUCKET

@pytest.fixture
def s3_prefix():
    return test_constant.S3_PREFIX

@pytest.fixture
def aws_region():
    return test_constant.AWS_REGION
