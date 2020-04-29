from s3_helper import S3Ref, download, upload
import json
from io import StringIO

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    This function adds a sequential id to each record in the input manifest.
    """
    s3_input_uri = event['ManifestS3Uri']
    s3_input = S3Ref.from_uri(s3_input_uri)

    inp_file = download(s3_input)
    logger.info("Downloaded file from {} to {}".format(s3_input_uri, inp_file))

    out_file = StringIO()
    total = 0
    for processed_id_count, line in enumerate(inp_file):
       data = json.loads(line)
       data["id"] = processed_id_count
       out_file.write(json.dumps(data) + "\n")
       total += 1
    logger.info("Added id field to {} records".format(total))

    # Uploading back to the same location where we downloaded the file from.
    upload(out_file, s3_input)
    logger.info("Uploaded updated file from {} to {}".format(
        out_file, s3_input_uri))
    return event

