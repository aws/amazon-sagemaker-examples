import pandas as pd
import time
import uuid
import boto3
from urllib.parse import urlparse
import datetime
import json
import io
import numpy as np

def remove_underrepresented_classes(features, labels, thresh=0.0005):
    """Removes classes when number of datapoints fraction is below a threshold."""
    total_count = labels.shape[0]
    unique, counts = np.unique(labels, return_counts=True)
    ratios = counts.astype('float') / total_count
    vals_and_ratios = dict(zip(unique, ratios))
    print('Unique classes and their ratio of total: %s' % vals_and_ratios)
    keep = [vals_and_ratios[v] >= thresh for v in labels]
    return features[keep], labels[np.array(keep)]

def safe_std(values):
    """Remove zero std values for ones."""
    return np.array([val if val != 0.0 else 1.0 for val in values])

def classification_to_bandit_problem(contexts, labels, num_actions=None):
    """Normalize contexts and encode deterministic rewards."""
    if num_actions is None:
        num_actions = np.max(labels) + 1
    num_contexts = contexts.shape[0]

    # Due to random subsampling in small problems, some features may be constant
    sstd = safe_std(np.std(contexts, axis=0, keepdims=True)[0, :])

    # Normalize features
    contexts = ((contexts - np.mean(contexts, axis=0, keepdims=True)) / sstd)

    # One hot encode labels as rewards
    rewards = np.zeros((num_contexts, num_actions))
    rewards[np.arange(num_contexts), labels] = 1.0

    return contexts, rewards, (np.ones(num_contexts), labels)


class StatlogSimApp():
    """
    A client application simulator using Statlog data.
    """
    def __init__(self, predictor):
        file_name = 'sim_app/shuttle.trn'
        self.num_actions = 7
        self.data_size = 43483
        
        with open(file_name, 'r') as f:
            data = np.loadtxt(f)

        # Shuffle data
        np.random.shuffle(data)

        # Last column is label, rest are features
        contexts = data[:, :-1]
        labels = data[:, -1].astype(int) - 1  # convert to 0 based index

        context, labels = remove_underrepresented_classes(contexts, labels)
        self.context, self.labels, _ = classification_to_bandit_problem(
                                        context, labels, self.num_actions)
        self.opt_rewards = [1]
        
        self.opt_rewards = [1]
        
        self.rewards_buffer = []
        self.joined_data_buffer = []

    def choose_random_user(self):
        context_index = np.random.choice(self.context.shape[0])
        context = self.context[context_index]
        return context_index, context
    
    def get_reward(self, 
                   context_index, 
                   action, 
                   event_id, 
                   model_id, 
                   action_prob, 
                   sample_prob, 
                   local_mode):

        reward = 1 if self.labels[context_index][action-1] == 1 else 0

        if local_mode:
            json_blob = {"reward": reward,
                         "event_id": event_id,
                         "action": action,
                         "action_prob": action_prob,
                         "model_id": model_id,
                         "observation": self.context[context_index].tolist(),
                         "sample_prob": sample_prob}
            self.joined_data_buffer.append(json_blob)
        else:
            json_blob = {"reward": reward, "event_id": event_id}
            self.rewards_buffer.append(json_blob)
        
        return reward
    
    def clear_buffer(self):
        self.rewards_buffer.clear()
        self.joined_data_buffer.clear()
