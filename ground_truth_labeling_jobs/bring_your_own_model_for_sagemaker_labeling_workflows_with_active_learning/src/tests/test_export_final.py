import boto3
from moto import mock_s3
from Output.export_final import lambda_handler

@mock_s3
def test_export_final():
    manifest_content = b'{"source": "tech now mindy kaling at sxsw", "category": 1, "category-metadata": {"confidence": 1.0, "human-annotated": "yes", "type": "groundtruth/text-classification"}}'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=manifest_content)
    s3r.create_bucket(Bucket='output_bucket')

    event = {
              'ManifestS3Uri': 's3://source_bucket/input.manifest',
              'FinalOutputS3Uri': 's3://output_bucket/'
            }

    output = lambda_handler(event, {})

    intermediate_body = s3r.Object('output_bucket', 'final_output.manifest').get()['Body'].read()
    assert intermediate_body == manifest_content

    assert output == "s3://output_bucket/final_output.manifest"
