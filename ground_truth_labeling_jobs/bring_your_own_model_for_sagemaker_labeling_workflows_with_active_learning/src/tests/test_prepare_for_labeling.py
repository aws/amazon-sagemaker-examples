from moto import mock_s3

from Labeling.prepare_for_labeling import lambda_handler

@mock_s3
def test_prepare_for_labeling(monkeypatch):
    def mock_copy(*args, **kwargs):
        source = args[0]
        dest = args[1]
        query = args[2]
        print("Copy mocked out source {} dest {} query {}".format(source, dest, query))
        return

    from Labeling import prepare_for_labeling
    monkeypatch.setattr(prepare_for_labeling, "copy_with_query", mock_copy)

    event = {
        'LabelingJobNamePrefix': 'jobprefix',
        'input_total': 10000,
        'human_label_done_count': 1000,
        'IntermediateFolderUri': 's3://output/',
        'LabelAttributeName': 'category',
        'ManifestS3Uri': 's3//input/input.manifest'
    }

    output = lambda_handler(event, {})

    assert output['human_input_s3_uri'] == 's3://output/human_input.manifest'
    assert output['labeling_job_name'].startswith('jobprefix')
    assert output['labeling_job_output_uri'].startswith('s3://output/labeling-job')

