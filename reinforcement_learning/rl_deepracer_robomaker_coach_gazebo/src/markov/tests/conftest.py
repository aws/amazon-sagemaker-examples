import pytest
from markov.tests import test_constant

@pytest.fixture
def aws_region():
    return test_constant.AWS_REGION

@pytest.fixture
def model_metadata_s3_key():
    return test_constant.MODEL_METADATA_S3_KEY

@pytest.fixture
def reward_function_s3_source():
    return test_constant.REWARD_FUNCTION_S3_SOURCE

@pytest.fixture
def s3_bucket():
    return test_constant.S3_BUCKET

@pytest.fixture
def s3_prefix():
    return test_constant.S3_PREFIX
