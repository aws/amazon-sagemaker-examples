from statlog_sim_app import remove_underrepresented_classes, classification_to_bandit_problem
import numpy as np
import pandas as pd
import boto3
from src.io_utils import parse_s3_uri

def prepare_statlog_warm_start_data(data_file, batch_size):
    """
    Generate a batch of experiences for warm starting the policy.
    """
    num_actions = 7
    joined_data_buffer = []

    with open(data_file, 'r') as f:
        data = np.loadtxt(f)

    # Shuffle data
    np.random.shuffle(data)

    # Last column is label, rest are features
    contexts = data[:, :-1]
    labels = data[:, -1].astype(int) - 1  # convert to 0 based index

    context, labels = remove_underrepresented_classes(contexts, labels)
    statlog_context, statlog_labels, _ = classification_to_bandit_problem(
                                    context, labels, num_actions)

    for i in range(0, batch_size):
        context_index_i = np.random.choice(statlog_context.shape[0])
        context_i = statlog_context[context_index_i]
        action = np.random.choice(num_actions) + 1 #random action
        action_prob = 1 / num_actions # probability of picking a random action
        reward = 1 if statlog_labels[context_index_i][action-1] == 1 else 0

        json_blob = {"reward": reward,
                    "event_id": 'not-apply-to-warm-start',
                    "action": action,
                    "action_prob": action_prob,
                    "model_id": 'not-apply-to-warm-start',
                    "observation": context_i.tolist(),
                    "sample_prob": np.random.uniform(0.0, 1.0)}

        joined_data_buffer.append(json_blob)

    return joined_data_buffer

def download_historical_data_from_s3(data_s3_prefix):
    """Download the warm start data from S3."""
    s3_client = boto3.client('s3')
    bucket, prefix, _ = parse_s3_uri(data_s3_prefix)

    results = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = results.get('Contents')
    key = contents[0].get('Key')
    
    data_file_name = 'statlog_warm_start.data'
    s3_client.download_file(bucket, key, data_file_name)

def evaluate_historical_data(data_file):
    """Calculate policy value of the logged policy."""
    # Assume logged data comes from same policy 
    # so no need for counterfactual analysis
    offline_data = pd.read_csv(data_file, sep=",")
    offline_data_mean = offline_data['reward'].mean()
    offline_data_cost = 1 - offline_data_mean
    offline_data_cost
    return offline_data_cost
