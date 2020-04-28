from functools import partial
from s3_helper import S3Ref, get_count_with_query

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    This function returns the counts of the labeling job records
       - input_total : total records in the manifest.
       - human_label : total records labeled by human.
       - auto_label : total records auto labeled.
       - unlabeled : count of records not yet labeled.
       - human_label_percentage : percentage of records labeled by humans.
    """
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    s3_input_uri = meta_data['IntermediateManifestS3Uri']

    source = S3Ref.from_uri(s3_input_uri)
    manifest_count = partial(get_count_with_query, source)
    logger.info("Getting counts from {}".format(s3_input_uri))
    total_query = "select count(*) from s3object s"
    human_labeled_query = """select count(*) from s3object[*] s where s."{}-metadata"."human-annotated" IN ('yes')""".format(
        label_attribute_name)
    auto_labeled_query = """select count(*) from s3object[*] s where s."{}-metadata"."human-annotated" IN ('no')""".format(
        label_attribute_name)

    manifest_size = manifest_count(total_query)
    human_labeled_count = manifest_count(human_labeled_query)
    auto_labeled_count = manifest_count(auto_labeled_query)
    unlabeled_count = manifest_size - (auto_labeled_count + human_labeled_count)
    human_label_percentage = int(human_labeled_count * 100.0 / manifest_size)
    counts = {
        "input_total" : manifest_size,
        "human_label" : human_labeled_count,
        "auto_label" : auto_labeled_count,
        "unlabeled" : unlabeled_count,
        "human_label_percentage" : human_label_percentage
    }

    # update the validation set count from previous iteration if present.
    if "counts" in meta_data and "validation" in meta_data["counts"]:
       counts["validation"] = meta_data["counts"]["validation"]
    else:
       counts["validation"] = 0

    logger.info("Counts computed {} ".format(str(counts)))
    return counts
