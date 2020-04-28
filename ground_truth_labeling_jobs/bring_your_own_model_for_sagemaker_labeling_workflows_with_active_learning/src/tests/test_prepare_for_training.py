from moto import mock_s3
import boto3
from io import StringIO
import string_helper
from ActiveLearning.prepare_for_training import lambda_handler

@mock_s3
def test_prepare_for_training(monkeypatch):

    def mock_copy(*args, **kwargs):
        source = args[0]
        dest = args[1]
        query = args[2]
        transform = args[3]
        mock_input = StringIO()
        mock_input.write('{"source": "This is a dog.", "id": 0}\n')
        mock_input.write('{"source": "This is a message about cat", "id": 1}\n')
        mock_input.seek(0)
        augmented_input = transform(mock_input)
        s3r.Object('output', 'active-learning-0ypM8t7c/training_input.manifest').put(Body=augmented_input.getvalue().encode())
        print("Copy with transform mocked out source {} dest {} query {}".format(source, dest, query))
        return mock_input

    def mock_download_query(*args, **kwargs):
        mock_validation = StringIO()
        mock_validation.write('{"id": 0}\n')
        mock_validation.seek(0)
        return mock_validation

    from ActiveLearning import prepare_for_training
    monkeypatch.setattr(prepare_for_training, "copy_with_query_and_transform", mock_copy)
    monkeypatch.setattr(prepare_for_training, "download_with_query", mock_download_query)
    monkeypatch.setattr(string_helper, "generate_random_string", lambda: "0ypM8t7c")
    event = {
        'LabelingJobNamePrefix': 'job-prefix',
        'LabelAttributeName': 'category',
        'ManifestS3Uri': 's3://input/input.manifest',
        'meta_data' : {
            'IntermediateFolderUri': 's3://output/',
            'ValidationS3Uri': 's3://input/validation.manifest',
            'training_config' : {
                'TrainingJobName' : 'job-name',
                'S3OutputPath' : 's3://output/'
            }
        }
    }

    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='input')
    s3r.create_bucket(Bucket='output')
    input = b'{"source": "This is a dog.", "id": 0}\n{"source": "This is a message about cat", "id": 1}\n'
    s3r.Object('input', 'input.manifest').put(Body=input)
    validation_input = b'{"source": "This is a dog.", "id": 0}\n'
    s3r.Object('input', 'validation.manifest').put(Body=validation_input)

    output = lambda_handler(event, {})

    training_input = s3r.Object('output', 'active-learning-0ypM8t7c/training_input.manifest').get()['Body'].read()
    expected_input = b'{"source": "This is a message about cat", "id": 1}\n'

    assert output['TrainingJobName'].startswith('job-prefix')
    assert output['trainS3Uri'] == 's3://output/active-learning-0ypM8t7c/training_input.manifest'
    assert output['ResourceConfig'] is not None
    assert output['AlgorithmSpecification'] is not None
    assert output['HyperParameters'] is not None
    assert output['S3OutputPath'] == 's3://output/active-learning-0ypM8t7c/'
    assert output['AttributeNames'] == ["source", "category"]
    assert training_input == expected_input

