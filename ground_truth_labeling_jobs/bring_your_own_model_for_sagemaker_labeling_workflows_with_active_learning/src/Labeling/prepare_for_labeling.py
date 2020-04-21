from s3_helper import S3Ref, copy_with_query
from string_helper import generate_job_id_and_s3_path

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_unlabeled_subset_count(input_total, human_label_done_count):
    """
    We want to label 20% of the dataset with humans.
    10% for training set and 10% for validation set.
    """
    unlabeled_subset_count = int(input_total/5) - human_label_done_count
    return unlabeled_subset_count

def lambda_handler(event, context):
    """
    Creates necessary input parameters for the first human labeling job so that after the job
    is complete 20% of the entire data is labelled.
    """
    job_name_prefix = event["LabelingJobNamePrefix"]
    input_total = event["input_total"]
    human_label_done_count = event["human_label_done_count"]
    intermediate_folder_uri = event["IntermediateFolderUri"]
    label_attribute_name = event['LabelAttributeName']
    s3_input_uri = event['ManifestS3Uri']

    unlabeled_subset_count = get_unlabeled_subset_count(input_total, human_label_done_count)

    source = S3Ref.from_uri(s3_input_uri)
    dest = S3Ref.from_uri(intermediate_folder_uri + "human_input.manifest")
    unlabeled_query = """select * from s3object[*] s where s."{}" is missing LIMIT {}""".format(
        label_attribute_name, unlabeled_subset_count)
    copy_with_query(source, dest, unlabeled_query)
    human_input_s3_uri = dest.get_uri()
    logging.info("Copied {} unlabled objects from {} to {}".format(
        unlabeled_subset_count, s3_input_uri, human_input_s3_uri))
    labeling_job_name, labeling_job_output_uri = generate_job_id_and_s3_path(
        job_name_prefix, intermediate_folder_uri, "labeling-job")

    return {
        "human_input_s3_uri": human_input_s3_uri,
        "labeling_job_name": labeling_job_name,
        "labeling_job_output_uri": labeling_job_output_uri,
    }
