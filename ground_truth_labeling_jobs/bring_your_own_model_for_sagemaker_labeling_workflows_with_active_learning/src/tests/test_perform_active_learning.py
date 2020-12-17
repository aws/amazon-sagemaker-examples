import boto3
from moto import mock_s3
from io import StringIO
from ActiveLearning.perform_active_learning import lambda_handler

@mock_s3
def test_peform_active_learning(monkeypatch):
    def mock_download_with_query(*args, **kwargs):
        output = StringIO()
        output.write('{"label":"dog"}\n')
        output.write('{"label":"cat"}\n')
        output.seek(0)
        return output

    class MockSimpleActiveLearning:
        def __init__(self, job_name, label_category_name,
                  label_names, max_selections):
            pass

        def autoannotate(self, predictions, sources):
            return ['{"source": "This is a message about cat", "category": 1, "category-metadata": {"confidence": 1.0, "human-annotated": "no", "type": "groundtruth/text-classification"}}']

        def select_for_labeling(self, predictions, autoannotations):
            return [0]

    from ActiveLearning import perform_active_learning
    monkeypatch.setattr(perform_active_learning, "download_with_query", mock_download_with_query)
    monkeypatch.setattr(perform_active_learning, "SimpleActiveLearning", MockSimpleActiveLearning)

    event = {
        'LabelCategoryConfigS3Uri': 's3://input/labels.json',
        'LabelingJobNamePrefix': 'job-prefix',
        'LabelAttributeName': 'Animal',
        'meta_data' : {
            'IntermediateFolderUri': 's3://output/',
            'UnlabeledS3Uri': 's3://input/unlabeled.manifest',
            'transform_config': {
               'S3OutputPath': 's3://input/'
            },
            'counts' : {
                'input_total' : 10000
            }
        }
    }
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='input')
    batch_tranform_input = b'{"source": "This is a dog.", "id": 0}\n{"source": "This is a message about cat", "id": 1}\n'
    s3r.Object('input', 'unlabeled.manifest').put(Body=batch_tranform_input)
    batch_tranform_output = b'{"id": 0, "prob": [0.6, 0.4], "label": ["__label__0", "__label__1"]}\n{"id": 1, "prob": [0.9, 0.1], "label": ["__label__1", "__label__0"]}\n'
    s3r.Object('input', 'unlabeled.manifest.out').put(Body=batch_tranform_output)

    output = lambda_handler(event, {})

    assert output['autoannotations'] == "s3://input/autoannotated.manifest"
    assert output['selections_s3_uri'] == 's3://input/selection.manifest'
    assert output['selected_job_name'].startswith('job-prefix')
    assert output['selected_job_output_uri'].startswith('s3://output/active-learning')
    assert output['counts']['autoannotated'] == 1
    assert output['counts']['selected'] == 1
