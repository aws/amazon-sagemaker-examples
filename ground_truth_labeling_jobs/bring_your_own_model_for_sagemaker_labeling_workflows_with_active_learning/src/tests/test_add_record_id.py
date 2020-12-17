import boto3
from moto import mock_s3

from Bootstrap.add_record_id import lambda_handler

@mock_s3
def test_add_record_id():
    manifest_content = b'{"source":"Fed revises guidelines sending stocks up."}\n{"source": "Review Guardians of the Galaxy"}'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=manifest_content)

    event = {
              'ManifestS3Uri': 's3://source_bucket/input.manifest',
            }

    output = lambda_handler(event, {})

    manifest_content_with_id = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0}\n{"source": "Review Guardians of the Galaxy", "id": 1}\n'
    updated_body = s3r.Object('source_bucket', 'input.manifest').get()['Body'].read()

    assert updated_body == manifest_content_with_id

    assert output['ManifestS3Uri'] == "s3://source_bucket/input.manifest"
