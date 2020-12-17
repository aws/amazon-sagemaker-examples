from moto import mock_s3

from ActiveLearning.create_validation_set import lambda_handler

@mock_s3
def test_prepare_for_labeling(monkeypatch):
    def mock_copy(*args, **kwargs):
        source = args[0]
        dest = args[1]
        query = args[2]
        print("Copy mocked out source {} dest {} query {}".format(source, dest, query))
        return

    from ActiveLearning import create_validation_set
    monkeypatch.setattr(create_validation_set, "copy_with_query", mock_copy)

    event = {
        'LabelAttributeName': 'category',
        'meta_data' : {
            'IntermediateManifestS3Uri': 's3://input/input.manifest',
            'counts' : {
                'input_total' : 10000
            }
        }
    }

    output = lambda_handler(event, {})

    assert output['counts']['validation'] == 1000
    assert output['ValidationS3Uri'] == 's3://input/validation_input.manifest'

