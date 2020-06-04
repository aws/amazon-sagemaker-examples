from s3_helper import S3Ref, copy

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    """
    This function is used to copy the final completed manifest to the output location.
    """
    s3_input_uri = event['ManifestS3Uri']
    source = S3Ref.from_uri(s3_input_uri)

    s3_output_uri = event['FinalOutputS3Uri'] + "final_output.manifest"
    dest = S3Ref.from_uri(s3_output_uri)

    copy(source, dest)
    logger.info("Copied s3 file from {} to {}".format(
        s3_input_uri, s3_output_uri))
    return dest.get_uri()

