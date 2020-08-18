from sagemaker import AutoML
import boto3
from time import gmtime, strftime, sleep
import json
import io
from urllib.parse import urlparse

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_candidates(top_n_candidates, job_name):
    # takes an autopilot job name, returns top candidates

    est = AutoML.attach(auto_ml_job_name = job_name)

    candidates = est.list_candidates(sort_by='FinalObjectiveMetricValue',
                                    sort_order='Descending',
                                    max_results=top_n_candidates)
    
    return est, candidates

def run_transform_jobs(bucket, prefix, est, candidates, input_data_transform):
    # takes a list of candidates, runs batch transform jobs
    inference_response_keys = ['predicted_label', 'probability']

    s3_transform_output_path = 's3://{}/{}/inference-results/'.format(bucket, prefix);

    transformers = []
        
    for candidate in candidates:
        model = est.create_model(name=candidate['CandidateName'],
                                    candidate=candidate,
                                    inference_response_keys=inference_response_keys)

        output_path = s3_transform_output_path + candidate['CandidateName'] +'/'

        transformers.append(
            model.transformer(instance_count=1, 
                              instance_type='ml.m5.xlarge',
                              assemble_with='Line',
                              output_path=output_path))

    # run the jobs
    for transformer in transformers:
        transformer.transform(data=input_data_transform, split_type='Line', content_type='text/csv', wait=False)
        print("Starting transform job {}".format(transformer._current_job_name))
    
    return transformers


def wait_until_completion(transformers):
    # takes a list of transformers, waits until the last job is completed
    
    sm = boto3.client('sagemaker')
    
    pending_complete = True
    
    while pending_complete:
        pending_complete = False
        num_transform_jobs = len(transformers)
        for transformer in transformers:
            desc = sm.describe_transform_job(TransformJobName=transformer._current_job_name)
            if desc['TransformJobStatus'] not in ['Failed', 'Completed']:
                pending_complete = True
            else:
                num_transform_jobs -= 1
        print("{} out of {} transform jobs are running.".format(num_transform_jobs, len(transformers)))
        sleep(30)

    for transformer in transformers:
        desc = sm.describe_transform_job(TransformJobName=transformer._current_job_name)
        print("Transform job '{}' finished with status {}".format(transformer._current_job_name, desc['TransformJobStatus']))
    
    return

def get_csv_from_s3(s3uri, file_name):
    parsed_url = urlparse(s3uri)
    bucket_name = parsed_url.netloc
    prefix = parsed_url.path[1:].strip('/')
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket_name, '{}/{}'.format(prefix, file_name))
    return obj.get()["Body"].read().decode('utf-8')    


def get_predictions(transformers, test_file):
    # takes completed transform jobs, gets data from s3
    predictions = []
    
    for transformer in transformers:
        print(transformer.output_path)
        pred_csv = get_csv_from_s3(transformer.output_path, '{}.out'.format(test_file))
        predictions.append(pd.read_csv(io.StringIO(pred_csv), header=None))
    
    return predictions

def get_roc_curve(predictions, candidates, labels):
    # takes predicted and actuals, plots an ROC curve
    fpr_tpr = []
    for prediction in predictions:
        fpr, tpr, _ = roc_curve(labels, prediction.loc[:,1])
        fpr_tpr.append(fpr)
        fpr_tpr.append(tpr)

    plt.figure(num=None, figsize=(16, 9), dpi=160, facecolor='w', edgecolor='k')
    plt.plot(*fpr_tpr)
    plt.legend([candidate['CandidateName'] for candidate in candidates], loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    
    return

def get_precision_recall(predictions, candidates, labels):
    # takes predicted and actuals, plots a precision recall curve
    precision_recall = []
    for prediction in predictions:
        precision, recall, _ = precision_recall_curve(labels, prediction.loc[:,1])
        precision_recall.append(recall)
        precision_recall.append(precision)

    plt.figure(num=None, figsize=(16, 9), dpi=160, facecolor='w', edgecolor='k')
    plt.plot(*precision_recall)
    plt.legend([candidate['CandidateName'] for candidate in candidates], loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()
    return

def get_candidate_for_precision_at_recall(predictions, candidates, labels, target_min_precision = 0.75):
    # takes best candidates, returns the best for recall at a given prediction level
    
    best_recall = 0
    best_candidate_idx = -1
    best_candidate_threshold = -1
    candidate_idx = 0
    for prediction in predictions:
        precision, recall, thresholds = precision_recall_curve(labels, prediction.loc[:,1])
        threshold_idx = np.argmax(precision>=target_min_precision)
        if recall[threshold_idx] > best_recall:
            best_recall = recall[threshold_idx]
            best_candidate_threshold = thresholds[threshold_idx]
            best_candidate_idx = candidate_idx
        candidate_idx += 1

    print("Best Candidate Name: {}".format(candidates[best_candidate_idx]['CandidateName']))
    print("Best Candidate Threshold (Operation Point): {}".format(best_candidate_threshold))
    print("Best Candidate Recall: {}".format(best_recall))
    
    return