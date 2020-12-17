from moto import mock_s3
import boto3
from io import StringIO

from ActiveLearning.prepare_for_inference import lambda_handler

@mock_s3
def test_prepare_for_inference(monkeypatch):
    '''
       There are only 2 records in the input.manifest and they both are sent to batch transform.
    '''
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
        s3r.Object('output', 'unlabeled.manifest').put(Body=augmented_input.getvalue().encode())
        print("Copy with transform mocked out source {} dest {} query {}".format(source, dest, query))
        return mock_input


    from ActiveLearning import prepare_for_inference
    monkeypatch.setattr(prepare_for_inference, "copy_with_query_and_transform", mock_copy)

    event = {
        'LabelAttributeName': 'category',
        'meta_data' : {
            'IntermediateManifestS3Uri': 's3://input/input.manifest',
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
    output = lambda_handler(event, {})

    batch_transform_input = s3r.Object('output', 'unlabeled.manifest').get()['Body'].read()
    expected_input = b'{"source": "This is a dog.", "id": 0, "k": 1000000}\n{"source": "This is a message about cat", "id": 1, "k": 1000000}\n'


    assert output['UnlabeledS3Uri'] == 's3://output/unlabeled.manifest'
    assert output['transform_config'] == {
        'TransformJobName' : 'job-name',
        'ModelName' : 'job-name',
        'S3OutputPath': 's3://output/'
    }
    assert batch_transform_input == expected_input

