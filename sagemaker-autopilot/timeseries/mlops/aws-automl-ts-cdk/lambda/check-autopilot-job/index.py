import boto3
import os
from json import loads, dumps
from datetime import datetime, timedelta

sagemaker = boto3.client('sagemaker')

def handler(event, context):
    job_name = event['AutoMLJobName']
    
    full_response = sagemaker.describe_auto_ml_job_v2(
        AutoMLJobName = job_name
    )
    
    job_status = full_response["AutoMLJobStatus"]
    
    # These values are appering only at the end of training process.
    if "BestCandidate" in full_response:
        best_candidate = full_response["BestCandidate"]
        
        inference_container = {}
        if "InferenceContainers" in best_candidate:
            inference_container = best_candidate['InferenceContainers'][0]
        else:
            inference_container = 'N/A'
        
        # Take only info which we need further about Candidate
        best_candidate_info = {
            'CandidateName': best_candidate["CandidateName"],
            'InferenceContainer': inference_container
        }
    else:
        best_candidate_info = "No Best Candidate ready yet."
        
    # In case Training job will fail.
    failure_reason = full_response["FailureReason"] if "FailureReason" in full_response else "N/A"
    
    # Take only info which we need further.
    return_object = {
        'AutoMLJobName': job_name,
        'AutoMLJobStatus': job_status,
        'BestCandidate': best_candidate_info,
        'AutoMLFailureReason': failure_reason
    }
    
    return return_object