import json
from collections import OrderedDict
from s3_helper import S3Ref, download, upload

import logging
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def merge_manifests(full_input, partial_output):
    """
    This method merges the output from partial output manifest to the full input
    to create the complete manifest.
    """
    complete_manifest = OrderedDict()
    for line in full_input:
        data = json.loads(line)
        complete_manifest[data["id"]] = data
    logger.info("Loaded input manifest of size {} to memory.".format(
        len(complete_manifest)))

    for line in partial_output:
        data = json.loads(line)
        complete_manifest[data["id"]] = data
    logger.info("Updated partial output in memory.")
    return complete_manifest

def lambda_handler(event, context):
    """
    This function is used to merge partial outputs to the manifest.
    The result is uploaded to s3.
    """
    s3_input_uri = event['ManifestS3Uri']
    source = S3Ref.from_uri(s3_input_uri)
    full_input = download(source)

    s3_output_uri = event['OutputS3Uri']
    output = S3Ref.from_uri(s3_output_uri)
    partial_output = download(output)

    logger.info("Downloaded input and output manifests {}, {}".format(
        s3_input_uri, s3_output_uri))

    complete_manifest = merge_manifests(full_input, partial_output)
    #write complete manifest back to s3 bucket
    merged = StringIO()
    for line in complete_manifest.values():
        merged.write(json.dumps(line) + "\n")
    upload(merged, source)
    logger.info("Uploaded merged file to {}".format(source.get_uri()))
