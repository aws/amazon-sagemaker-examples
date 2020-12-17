import pytest
import boto3
from moto import mock_s3
from Bootstrap.copy_input_manifest import lambda_handler

@mock_s3
def test_copy_input_manifest():
    manifest_content = b'{"source":"Fed revises guidelines sending stocks up."}'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=manifest_content)
    s3r.create_bucket(Bucket='output_bucket')

    event = {
              'ManifestS3Uri': 's3://source_bucket/input.manifest', 
              'S3OutputPath': 's3://output_bucket/'
            }

    output = lambda_handler(event, {})

    intermediate_body = s3r.Object('output_bucket', 'intermediate/input.manifest').get()['Body'].read()
    assert intermediate_body == manifest_content

    assert output['IntermediateFolderUri'] == "s3://output_bucket/intermediate/"
    assert output['IntermediateManifestS3Uri'] == "s3://output_bucket/intermediate/input.manifest"


def test_copy_input_manifest_invalid_output():
    with pytest.raises(Exception, match=r"S3OutputPath should end with '/'"):
      event = {
                'ManifestS3Uri': 's3://source_bucket/input.manifest', 
                'S3OutputPath': 's3://output_bucket'
              }

      lambda_handler(event, {})
      
