"""Custom Model Monitoring script for Detecting Data Drift in NLP using SageMaker Model Monitor
"""

# Python Built-Ins:
from collections import defaultdict
import datetime
import json
import os
import traceback
from types import SimpleNamespace

# External Dependencies:
import numpy as np
import boto3
from sentence_transformers import SentenceTransformer
import numpy as np


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MMD(x, y, kernel):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3] #bandwidth range for multiscale kernel
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50] #bandwidth range for rbf kernel
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)

    return torch.mean(XX + YY - 2. * XY)


def get_environment():
    """Load configuration variables for SM Model Monitoring job

    See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-contract-inputs.html
    """
    try:
        with open("/opt/ml/config/processingjobconfig.json", "r") as conffile:
            defaults = json.loads(conffile.read())["Environment"]
    except Exception as e:
        traceback.print_exc()
        print("Unable to read environment vars from SM processing config file")
        defaults = {}

    return SimpleNamespace(
        dataset_format=os.environ.get("dataset_format", defaults.get("dataset_format")),
        dataset_source=os.environ.get(
            "dataset_source",
            defaults.get("dataset_source", "/opt/ml/processing/input/endpoint"),
        ),
        end_time=os.environ.get("end_time", defaults.get("end_time")),
        output_path=os.environ.get(
            "output_path",
            defaults.get("output_path", "/opt/ml/processing/resultdata"),
        ),
        publish_cloudwatch_metrics=os.environ.get(
            "publish_cloudwatch_metrics",
            defaults.get("publish_cloudwatch_metrics", "Enabled"),
        ),
        sagemaker_endpoint_name=os.environ.get(
            "sagemaker_endpoint_name",
            defaults.get("sagemaker_endpoint_name"),
        ),
        sagemaker_monitoring_schedule_name=os.environ.get(
            "sagemaker_monitoring_schedule_name",
            defaults.get("sagemaker_monitoring_schedule_name"),
        ),
        start_time=os.environ.get(
            "start_time", 
            defaults.get("start_time")),
        max_ratio_threshold=float(os.environ.get(
            "THRESHOLD", 
             defaults.get("THRESHOLD", "nan"))),
        bucket=os.environ.get(
            "bucket",
            defaults.get("bucket", "None")),
    )


def download_embeddings_file():
    
    env = get_environment()
    from s3fs.core import S3FileSystem
    s3 = S3FileSystem()
    
    key = 'sagemaker/nlp-model-monitor/embeddings/embeddings.npy'
    bucket = env.bucket
    print("S3 bucket name is",bucket)

    return np.load(s3.open('{}/{}'.format(bucket, key)))
    
if __name__=="__main__":

    env = get_environment()
    print(f"Starting evaluation with config\n{env}")

    print(f"Downloading Embedding File")
    
    #download BERT embedding file used for fine-tuning BertForSequenceClassification
    baseline_embedding_list = download_embeddings_file()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    sent_mmd_dict = {}
    violations = []
    
    total_record_count = 0  # Including error predictions that we can't read the response for
    error_record_count = 0
    counts = defaultdict(int)  # dict defaulting to 0 when unseen keys are requested
    for path, directories, filenames in os.walk(env.dataset_source):
        for filename in filter(lambda f: f.lower().endswith(".jsonl"), filenames):
            with open(os.path.join(path, filename), "r") as file:
                for entry in file:
                    total_record_count += 1
                    try:
                        response = json.loads(json.loads(entry)["captureData"]["endpointInput"]["data"])
                    except:
                        print("Inside Exception")
                        continue
                
                    record = list(response.values())
                    print(f"Input Sentence: {record}")
                    
                    for rec in record:
                        record_sentence_embedding = model.encode(rec)[None]
                        tensor_embedding1 = torch.from_numpy(record_sentence_embedding).to(device)
                        
                        mmd_score = 0
                        
                        for embed_item in baseline_embedding_list:
                            
                            tensor_embedding2 = torch.from_numpy(embed_item).to(device)

                            result = MMD(tensor_embedding1, tensor_embedding2, kernel="multiscale")
                            print(f"MMD result of X and Y is {result.item()}")
                            
                            mmd_score+= result
                            
                        mmd_score_avg = mmd_score/(len(baseline_embedding_list))
                        print(f"average mmd score: {mmd_score_avg}")
                        if mmd_score_avg > env.max_ratio_threshold:
                            error_record_count += 1
                            sent_mmd_dict[rec] = mmd_score_avg
                            violations.append({
                                    "sentence": rec,
                                    #"avg_mmd_score": mmd_score_avg,
                                    "feature_name": "sent_mmd_score",
                                    "constraint_check_type": "baseline_drift_check",
                                    "endpoint_name" : env.sagemaker_endpoint_name,
                                    "monitoring_schedule_name": env.sagemaker_monitoring_schedule_name
                                })
        
    print("Checking for constraint violations...")
    print(f"Violations: {violations if len(violations) else 'None'}")

    print("Writing violations file...")
    with open(os.path.join(env.output_path, "constraints_violations.json"), "w") as outfile:
        outfile.write(json.dumps(
            { "violations": violations },
            indent=4,
        ))
    
    print("Writing overall status output...")
    with open("/opt/ml/output/message", "w") as outfile:
        if len(violations):
            msg = "CompletedWithViolations"
        else:
            msg = "Completed: Job completed successfully with no violations."
        outfile.write(msg)
        print(msg)
    '''
    if True:
    #if env.publish_cloudwatch_metrics:
        print("Writing CloudWatch metrics...")
        with open("/opt/ml/output/metrics/cloudwatch/cloudwatch_metrics.jsonl", "a+") as outfile:
            # One metric per line (JSONLines list of dictionaries)
            # Remember these metrics are aggregated in graphs, so we report them as statistics on our dataset
            outfile.write(json.dumps(
            { "violations": str(len(violations)) },
            indent=4,
            ))
    '''
    print("Done")