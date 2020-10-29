import json
from functools import partial

from io import StringIO
from s3_helper import S3Ref, copy_with_query_and_transform, download_with_query, create_ref_at_parent_key
from string_helper import generate_job_id_and_s3_path

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def remove_by_ids(s3_blacklist_uri, label_attribute_name, manifest_file):
    """
    helper method to remove selected id in the given input file.
    This is used to create a training set which has no elements from the given validation set.
    """
    logger.info("Remove validation set ids from training data.")
    blacklist = S3Ref.from_uri(s3_blacklist_uri)
    validation_id_query = """select s."id" from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes')""".format(
        label_attribute_name)
    validation_id_file = download_with_query(blacklist, validation_id_query)
    validation_ids = set()

    for line in validation_id_file:
        data = json.loads(line)
        validation_ids.add(data["id"])

    training_only_file = StringIO()
    training_set_size = 0
    for line in manifest_file:
        data = json.loads(line)
        if data["id"] not in validation_ids:
            training_set_size += 1
            training_only_file.write(json.dumps(data) + "\n")
    logger.info("Remove ids complete. training set size = {} Validation set size = {}".format(
             training_set_size, len(validation_ids)))
    return training_only_file

class TrainingJobParameters:

   def __init__(self, event, training_folder_uri):
       self.event = event
       self.training_folder_uri = training_folder_uri

   @property
   def attribute_names(self):
    """
    attribute names to be parsed from the manifest file during training.
    """
    label_attribute_name = self.event['LabelAttributeName']
    input_mode = "source"
    return [input_mode, label_attribute_name]

   @property
   def training_input(self):
    """
    Generates the training input in an s3 location and returns the s3 uri.
    """
    label_attribute_name = self.event['LabelAttributeName']
    s3_input_uri = self.event['ManifestS3Uri']
    meta_data = self.event['meta_data']

    source = S3Ref.from_uri(s3_input_uri)
    dest = S3Ref.from_uri(self.training_folder_uri + "training_input.manifest")
    logger.info("Creating training input at {} from human labeled data.".format(
        dest.get_uri()))
    removeValidationIds = partial(remove_by_ids, meta_data['ValidationS3Uri'],
                            label_attribute_name)
    training_labeled_query = """select * from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes')""".format(
        label_attribute_name)
    copy_with_query_and_transform(source, dest, training_labeled_query, removeValidationIds)
    logger.info("Uploaded training input at {}.".format(dest.get_uri()))
    return dest.get_uri()

   @property
   def resource_config(self):
    """
    configure the instance where training will be run.
    """
    return {
        "InstanceCount": 1,
        "InstanceType": "ml.c5.2xlarge",
        "VolumeSizeInGB": 60
    }

   @property
   def algorithm_specification(self):
    """
     configure the docker container uri for the training algorithm.
    """
    return {
        # This assumes we are running in us-east-1 (IAD).
        # Refer to this doc to tweak this model if you run it in other regions.
        # https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
        "TrainingImage": "811284229777.dkr.ecr.us-east-1.amazonaws.com/blazingtext:latest",
        "TrainingInputMode":"Pipe"
    }

   @property
   def hyper_parameters(self):
    """
      configure hyper parameters used for training.
    """
    return {
        "early_stopping":"True",
        "epochs":"20",
        "learning_rate":"0.05",
        "min_count":"5",
        "min_epochs":"1",
        "mode":"supervised",
        "patience":"5",
        "vector_dim":"20",
        "word_ngrams":"2"
    }

def lambda_handler(event, context):
    """
    This function sets up all the input parameters required for the training job.
    """
    training_job_name_prefix = event['LabelingJobNamePrefix']
    intermediate_folder_uri = event["meta_data"]["IntermediateFolderUri"]
    training_job_name, training_folder_uri = generate_job_id_and_s3_path(
        training_job_name_prefix, intermediate_folder_uri)
    training_job_parameters = TrainingJobParameters(event, training_folder_uri)

    return {
             "TrainingJobName": training_job_name,
             "trainS3Uri":training_job_parameters.training_input,
             "ResourceConfig":training_job_parameters.resource_config,
             'AlgorithmSpecification':training_job_parameters.algorithm_specification,
             "HyperParameters":training_job_parameters.hyper_parameters,
             "S3OutputPath": training_job_parameters.training_folder_uri,
             "AttributeNames": training_job_parameters.attribute_names
    }
