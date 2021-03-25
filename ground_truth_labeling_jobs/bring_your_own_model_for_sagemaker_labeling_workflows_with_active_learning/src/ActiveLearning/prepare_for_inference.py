import json

from io import StringIO
from s3_helper import S3Ref, copy_with_query_and_transform, create_ref_at_parent_key

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def augment_inference_input(inference_raw):
    """
     The inference manifest needs to be augmented with a value 'k' so that blazing text
     produces all probabilities instead of just the top match.
    """
    augmented_inference = StringIO()
    for line in inference_raw:
       infer_dict = json.loads(line)
       # Note: This number should ideally be equal to the number of classes.
       # But using a big number, produces the same result.
       infer_dict['k'] = 1000000
       augmented_inference.write(json.dumps(infer_dict) + "\n")
    logger.info("Augmented inference data by adding 'k' to each line.")
    return augmented_inference

def create_tranform_config(training_config):
    """
     Transform config specifies input parameters for the transform job.
    """
    return {
        # We reuse the training job name for the model name and corresponding
        # transform job name.
        'TransformJobName' : training_config['TrainingJobName'],
        'ModelName' : training_config['TrainingJobName'],
        'S3OutputPath' : training_config['S3OutputPath']
    }

def lambda_handler(event, context):
    """
    This function generates auto annotations and performs active learning.
    """
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    s3_input_uri = meta_data['IntermediateManifestS3Uri']

    transform_config = create_tranform_config(meta_data['training_config'])

    source = S3Ref.from_uri(s3_input_uri)
    dest = S3Ref.from_uri(transform_config['S3OutputPath'] + "unlabeled.manifest")

    logger.info("Creating inference output from unlabeled subset of input {}.".format(
        s3_input_uri))
    SQL_UNLABELED = """select * from s3object[*] s where s."{}" is missing """
    unlabeled_query = SQL_UNLABELED.format(label_attribute_name)
    copy_with_query_and_transform(
        source, dest, unlabeled_query, augment_inference_input)
    meta_data['UnlabeledS3Uri'] = dest.get_uri()
    logger.info("Uploaded unlabeled manifest for inference to {}.".format(
        dest.get_uri()))

    meta_data['transform_config'] = transform_config
    return meta_data
