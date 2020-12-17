import boto3
from moto import mock_s3

from MetaData.get_counts import lambda_handler

@mock_s3
def test_get_counts_nothing_labeled(monkeypatch):
    '''
     This test requires monkeypatching because select_object_content is not implemented by moto.
     https://github.com/spulec/moto/blob/master/IMPLEMENTATION_COVERAGE.md
    '''
    manifest_content = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0}\n{"source": "Review Guardians of the Galaxy", "id": 1}\n'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=manifest_content)

    def mock_count(*args, **kwargs):
        key = args[0].key
        query = args[1]
        if "input.manifest" == key and 'select count(*) from s3object s' == query:
            return 2
        return 0

    from MetaData import get_counts
    monkeypatch.setattr(get_counts, "get_count_with_query", mock_count)

    event = {
              'LabelAttributeName': 'category',
              'meta_data': {
                  "IntermediateManifestS3Uri": 's3://source_bucket/input.manifest'
              }
            }

    expected_counts = {
        "input_total" : 2,
        "human_label" : 0,
        "auto_label" : 0,
        "unlabeled" : 2,
        "human_label_percentage" : 0,
        "validation": 0
    }
    output = lambda_handler(event, {})

    assert output == expected_counts

@mock_s3
def test_get_counts_everything_labeled(monkeypatch):
    manifest_content = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0}\n{"source": "Review Guardians of the Galaxy", "id": 1}\n'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=manifest_content)

    def mock_count(*args, **kwargs):
        key = args[0].key
        query = args[1]
        if "input.manifest" == key and 'select count(*) from s3object s' == query:
            return 2000
        if "input.manifest" == key and '"human-annotated" IN (\'yes\')' in query:
            return 1000
        if "input.manifest" == key and '"human-annotated" IN (\'no\')' in query:
            return 1000
        return 0

    from MetaData import get_counts
    monkeypatch.setattr(get_counts, "get_count_with_query", mock_count)

    event = {
              'LabelAttributeName': 'category',
              'meta_data': {
                  "IntermediateManifestS3Uri": 's3://source_bucket/input.manifest',
                  "counts": {
                     "validation" : 500
                  }
              }
            }

    expected_counts = {
        "input_total" : 2000,
        "human_label" : 1000,
        "auto_label" : 1000,
        "unlabeled" : 0,
        "human_label_percentage" : 50,
        "validation": 500
    }
    output = lambda_handler(event, {})

    assert output == expected_counts
