import json

from s3_helper import S3Ref, download, download_with_query, upload, create_ref_at_parent_key
from string_helper import generate_job_id_and_s3_path
from io import StringIO

from ActiveLearning.helper import SimpleActiveLearning

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_label_names_from_s3(labels_s3_uri):
    """
     fetch the list of labels from a label s3 bucket.
    """
    labels_source = S3Ref.from_uri(labels_s3_uri)
    labeled_query = """SELECT label FROM S3Object[*].labels[*].label"""
    result_inp = download_with_query(labels_source, labeled_query)
    label_names = []
    for line in result_inp:
        label_data = json.loads(line)
        label_names.append(label_data['label'])
    return label_names


def get_sources(inference_input):
    """
     Load inference input as a python list
    """
    sources = []
    for line in inference_input:
        data = json.loads(line)
        sources.append(data)
    inference_input.seek(0)
    return sources

def get_predictions(inference_output):
    """
     Load inference output as a python list
    """
    predictions = []
    for line in inference_output:
      data = json.loads(line)
      prediction = {}
      for key, value in data.items():
          if key != "SageMakerOutput":
              prediction[key] = value
          else:
              if not isinstance(value, dict):
                 print("Error: Expected dictionary inside SageMakerOutput.")
              prediction.update(value)
      predictions.append(prediction)
    return predictions

def collect_inference_inputs(s3_input_uri):
    """
     collect information related to input to inference.
    """
    inference_input_s3_ref = S3Ref.from_uri(s3_input_uri)
    inference_input = download(inference_input_s3_ref)
    sources = get_sources(inference_input)
    logger.info("Collected {} inference inputs.".format(len(sources)))
    return inference_input_s3_ref, inference_input, sources

def collect_inference_outputs(inference_output_uri):
    """
     collect information related to output of inference.
    """
    sagemaker_output_file = "unlabeled.manifest.out"
    prediction_output_uri = inference_output_uri + sagemaker_output_file
    prediction_output_s3 = S3Ref.from_uri(prediction_output_uri)
    prediction_output = download(prediction_output_s3)
    predictions = get_predictions(prediction_output)
    logger.info("Collected {} inference outputs.".format(len(predictions)))
    return predictions

def write_auto_annotations(simple_al, sources, predictions, inference_input_s3_ref):
    """
     write auto annotations to s3
    """
    logger.info("Generating auto annotations where confidence is high.")
    auto_annotation_stream = StringIO()
    auto_annotations = simple_al.autoannotate(predictions, sources)
    for auto_annotation in auto_annotations:
        auto_annotation_stream.write(json.dumps(auto_annotation) + "\n")

    # Auto annotation.
    auto_dest = create_ref_at_parent_key(inference_input_s3_ref, "autoannotated.manifest")
    upload(auto_annotation_stream, auto_dest)
    logger.info("Uploaded autoannotations to {}.".format(auto_dest.get_uri()))
    return auto_dest.get_uri(), auto_annotations

def write_selector_file(simple_al, sources, predictions, inference_input_s3_ref, inference_input, auto_annotations):
    """
     write selector file to s3. This file is used to decide which records should be labeled by humans next.
    """
    logger.info("Selecting input for next manual annotation")
    selection_data = StringIO()
    selections = simple_al.select_for_labeling(predictions, auto_annotations)
    selections_set = set(selections)
    for line in inference_input:
       data = json.loads(line)
       if data["id"] in selections_set:
           selection_data.write(json.dumps(data) + "\n")
    inference_input.seek(0)
    selection_dest = create_ref_at_parent_key(
        inference_input_s3_ref, "selection.manifest")
    upload(selection_data, selection_dest)
    logger.info("Uploaded selections to {}.".format(selection_dest.get_uri()))
    return selection_dest.get_uri(), selections

def lambda_handler(event, context):
    """
    This function generates auto annotatations and performs active learning.
    - auto annotations generates machine labels for confident examples.
    - active learning selects for examples to be labeled by humans next.
    """
    labels_s3_uri = event['LabelCategoryConfigS3Uri']
    job_name_prefix = event['LabelingJobNamePrefix']
    job_name = "labeling-job/{}".format(job_name_prefix)
    label_attribute_name = event['LabelAttributeName']
    meta_data = event['meta_data']
    intermediate_folder_uri = meta_data["IntermediateFolderUri"]
    input_total = int(meta_data['counts']['input_total'])
    # Select maximum of 10% of the input total for next round of manual labeling.
    max_selections = input_total // 10
    # Handle corner case where integer division can lead us to 0 selections.
    if max_selections == 0:
      max_selections = input_total

    inference_input_s3_ref, inference_input, sources = \
               collect_inference_inputs(meta_data['UnlabeledS3Uri'])
    predictions = \
       collect_inference_outputs(meta_data['transform_config']['S3OutputPath'])
    label_names = get_label_names_from_s3(labels_s3_uri)
    logger.info("Collected {} label names.".format(len(label_names)))

    simple_al = SimpleActiveLearning(job_name, label_attribute_name,
                 label_names, max_selections)
    meta_data['autoannotations'], auto_annotations = write_auto_annotations(
        simple_al, sources, predictions, inference_input_s3_ref)
    meta_data['selections_s3_uri'], selections = write_selector_file(
        simple_al, sources, predictions, inference_input_s3_ref, inference_input, auto_annotations)
    meta_data['selected_job_name'], meta_data['selected_job_output_uri'] = generate_job_id_and_s3_path(
        job_name_prefix, intermediate_folder_uri)
    meta_data['counts']['autoannotated'] = len(auto_annotations)
    meta_data['counts']['selected'] = len(selections)
    return meta_data
