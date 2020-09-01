""" The script takes care of testing the functionality of markov/s3/model_metadata
parse_model_metadata"""

import pytest
import json
import os

from markov.s3.files.model_metadata import ModelMetadata
from markov.constants import SIMAPP_VERSION_1, SIMAPP_VERSION_2
from markov.architecture.constants import Input, NeuralNetwork

@pytest.fixture
def create_model_metadata_action_space():
    """Fixture function which creates the model metadata json file
    without the sensor or neural_network information

    Returns:
        String: File path for model_metadata json file created in environment for testing
    """
    model_metadata_path = "test_model_metadata_action_space.json"
    model_metadata = {
        "action_space": [
            {
                "steering_angle": 45,
                "speed": 0.8
            },
            {
                "steering_angle": -45,
                "speed": 0.8
            },
            {
                "steering_angle": 0,
                "speed": 0.8
            },
            {
                "steering_angle": 22.5,
                "speed": 0.8
            },
            {
                "steering_angle": -22.5,
                "speed": 0.8
            },
            {
                "steering_angle": 0,
                "speed": 0.4
            }
            ]
    }
    with open(model_metadata_path, 'w') as file:
        json.dump(model_metadata, file, indent=4)
    return model_metadata_path

@pytest.fixture
def create_model_metadata():
    """Fixture function which creates the model_metadata json file
    with the sensor and neural_network information

    Returns:
        String: File path for model_metadata json file created in environment for testing
    """
    model_metadata_path = "test_model_metadata.json"
    model_metadata = {
        "action_space": [
            {
                "steering_angle": -30,
                "speed": 0.6
            },
            {
                "steering_angle": -15,
                "speed": 0.6
            },
            {
                "steering_angle": 0,
                "speed": 0.6
            },
            {
                "steering_angle": 15,
                "speed": 0.6
            },
            {
                "steering_angle": 30,
                "speed": 0.6
            }
        ],
        "sensor": ["STEREO_CAMERAS"],
        "neural_network": "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    }
    with open(model_metadata_path, 'w') as file:
        json.dump(model_metadata, file, indent=4)
    return model_metadata_path

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_parse_model_metadata_only_action(create_model_metadata_action_space):
    """This function tests the functionality of parse_model_metadata function
    in markov/s3/model_metadata parse_model_metadata when we pass a model metadata file with only
    action space and no sensor or neural network information.

    Args:
        create_model_metadata_action_space (String): Gives the path for model metadata file for testing
    """
    sensor, network, simapp_version = ModelMetadata.parse_model_metadata(create_model_metadata_action_space)
    os.remove(create_model_metadata_action_space)
    assert sensor == [Input.OBSERVATION.value]
    assert network == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value
    assert simapp_version == SIMAPP_VERSION_1

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_parse_model_metadata(create_model_metadata):
    """This function tests the functionality of parse_model_metadata function
    in markov/s3/model_metadata parse_model_metadata when we pass a model metadata file with
    sensor and neural network information

    Args:
        create_model_metadata (String): Gives the path for model metadata file for testing
    """
    sensor, network, simapp_version = ModelMetadata.parse_model_metadata(create_model_metadata)
    os.remove(create_model_metadata)
    assert sensor == ["STEREO_CAMERAS"]
    assert network == "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    assert simapp_version == SIMAPP_VERSION_2

@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_parse_model_metadata_exception():
    """This function tests the functionality of parse_model_metadata function
    in markov/s3/model_metadata parse_model_metadata when an exception occurs
    """
    with pytest.raises(Exception, match=r".*Model metadata does not exist:.*"):
        sensor, network, simapp_version = ModelMetadata.parse_model_metadata("dummy_file.json")


@pytest.mark.robomaker
@pytest.mark.sagemaker
def test_load_model_metadata(s3_bucket, aws_region, model_metadata_s3_key):
    """This function checks the functionality of get_model_metadata_info in
    in markov/s3/model_metadata.py

    The function checks if model_metadata.json file is downloaded into the required directory.
    If the function fails, it will generate an exception which will call log_and_exit internally.
    Hence the test will fail.

    Args:
        s3_bucket (String): S3_BUCKET
        aws_region (String): AWS_REGION
        model_metadata_s3_key (String): MODEL_METADATA_S3_KEY
    """
    model_metadata_local_path = 'test_model_metadata.json'
    model_metadata = ModelMetadata(bucket=s3_bucket,
                                   s3_key=model_metadata_s3_key,
                                   region_name=aws_region,
                                   local_path=model_metadata_local_path)
    model_metadata.get_model_metadata_info()
    assert os.path.isfile(model_metadata_local_path)
    # Remove file downloaded
    if os.path.isfile(model_metadata_local_path):
        os.remove(model_metadata_local_path)
