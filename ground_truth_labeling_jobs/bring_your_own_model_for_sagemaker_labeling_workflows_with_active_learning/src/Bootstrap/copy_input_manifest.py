from s3_helper import S3Ref, get_content_size, copy

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def validate_output_path(s3_output_uri):
    """
    This method validates the input uri to make sure it ends with a "/" representing a s3 folder.
    """
    # The output path should end with a "/" to indicate a folder intead of a file.
    if len(s3_output_uri) == 0 or not s3_output_uri.endswith('/'):
        raise Exception("S3OutputPath should end with '/'.")

def lambda_handler(event, context):
    """
    This function does a copy of the input manifest to the a location within the specified output path
    after performing the following validations
        1. The output uri is not empty and ends with a '/'.
              This condition throws a exception.
        2. The input refers to a manifest file of size <= 80 MB.
              This condition records a warning in the log and allows the code to proceed.
    """
    logger.debug("event {} context {}".format(event, context))
    s3_input_uri = event['ManifestS3Uri']
    s3_output_uri = event['S3OutputPath']
    validate_output_path(s3_output_uri)

    source = S3Ref.from_uri(s3_input_uri)
    # Add a warning if the input file is too big.
    # These limits are due to limited runtime and memory in lambda.
    size = get_content_size(source)
    SIZE_MESSAGE=""""This tutorial was not tested for inputs greater than 80 MB (approx 200,000 objects). You are using a %d MB input manifest file."""
    if size > 80 * 1024 * 1024:
        logger.warn(SIZE_MESSAGE, size/1024/1024)

    # Final output folder of the labeling job
    output_folder = S3Ref.from_uri(s3_output_uri)

    # Create intermediate folder within the final output folder for saving
    # partially complete results.
    intermediate_folder_uri = s3_output_uri + 'intermediate/'
    intermediate_file_uri = intermediate_folder_uri + "input.manifest"

    # Copy original input to the intermediate s3 folder.
    dest = S3Ref.from_uri(intermediate_file_uri)
    logger.info("Copying s3 file from {} to {}".format(
        s3_input_uri, intermediate_file_uri))
    copy(source, dest)
    logger.info("Copied s3 file from {} to {}".format(
        s3_input_uri, intermediate_file_uri))

    return {
        "IntermediateFolderUri": intermediate_folder_uri,
        "IntermediateManifestS3Uri" : intermediate_file_uri
    }

