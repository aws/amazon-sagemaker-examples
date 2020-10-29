import boto3
from moto import mock_s3
from Output.export_partial import lambda_handler

@mock_s3
def test_export_partial():
    input_manifest_content = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0}\n{"source": "Review Guardians of the Galaxy", "id": 1}\n'
    output_manifest_content = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0, "category": 1, "category-metadata": {"confidence": 1.0, "human-annotated": "yes", "type": "groundtruth/text-classification"}}\n'
    s3r = boto3.resource('s3', region_name='us-east-1')
    s3r.create_bucket(Bucket='source_bucket')
    s3r.Object('source_bucket', 'input.manifest').put(Body=input_manifest_content)
    s3r.create_bucket(Bucket='output_bucket')
    s3r.Object('output_bucket', 'output.manifest').put(Body=output_manifest_content)

    event = {
              'ManifestS3Uri': 's3://source_bucket/input.manifest',
              'OutputS3Uri': 's3://output_bucket/output.manifest'
            }

    lambda_handler(event, {})

    expected_manifest_content = b'{"source": "Fed revises guidelines sending stocks up.", "id": 0, "category": 1, "category-metadata": {"confidence": 1.0, "human-annotated": "yes", "type": "groundtruth/text-classification"}}\n{"source": "Review Guardians of the Galaxy", "id": 1}\n'
    body = s3r.Object('source_bucket', 'input.manifest').get()['Body'].read()
    print(body)
    print(expected_manifest_content)
    assert body == expected_manifest_content

