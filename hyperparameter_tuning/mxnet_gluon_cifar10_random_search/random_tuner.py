import random
import time
import boto3
import re
import pandas as pd
import numpy as np


#################
# Hyperparameters
#################


class CategoricalParameter():
    '''
    Class for categorical hyperparameters.
    Takes one argument which is a list of possible hyperparameter values.
    '''
    def __init__(self, values):
        self.values = values
    def get_value(self):
        return random.choice(self.values)


class IntegerParameter():
    '''
    Class for integer hyperparameters.
    Takes two arguments: min_value and then max_value.
    '''
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def get_value(self):
        return random.randint(self.min_value, self.max_value)


class ContinuousParameter():
    '''
    Class for continuous hyperparameters.
    Takes two arguments: min_value and then max_value.
    '''
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
    def get_value(self):
        return random.uniform(self.min_value, self.max_value)


###############
# Random search
###############


def _get_random_hyperparameter_values(hyperparameters):
    '''
    Converts a dict using hyperparameter classes to a dict of hyperparameter values.
    '''
    hps = {}
    for hp, definition in hyperparameters.items():
        if isinstance(definition, (CategoricalParameter, IntegerParameter,
                                   ContinuousParameter)):
            hps[hp] = definition.get_value()
        else:
            hps[hp] = definition
    return hps


def random_search(train_fn,
                  hyperparameters,
                  base_name=None,
                  max_jobs=100,
                  max_parallel_jobs=100):
    '''
    Runs random search for hyperparameters.
    Takes in:
        train_fn: A function that kicks off a training job based on two positional arguments-
            job name and hyperparameter dictionary.  Note, wait must be set to False if using .fit()
        hyperparameters: A dictonary of hyperparameters defined with hyperparameter classes.
        base_name: Base name for training jobs.  Defaults to 'random-hp-<timestamp>'.
        max_jobs: Total number of training jobs to run.
        max_parallel_jobs: Most training jobs to run concurrently. This does not affect the quality
            of search, just helps stay under account service limits.
    Returns a dictionary of max_jobs job names with associated hyperparameter values.
    '''

    if base_name is None:
        base_name = 'random-hp-' + time.strftime('%Y-%m-%d-%H-%M-%S-%j', time.gmtime())

    client = boto3.client('sagemaker')
    jobs = {}
    running_jobs = {}
    for i in range(max_jobs):
        job = base_name + '-' + str(i)
        hps = _get_random_hyperparameter_values(hyperparameters)
        jobs[job] = hps.copy()
        train_fn(job, hps)
        running_jobs[job] = True
        while len(running_jobs) == max_parallel_jobs:
            for job in list(running_jobs):
                if client.describe_training_job(TrainingJobName=job)['TrainingJobStatus'] != 'InProgress':
                    running_jobs.pop(job)
            time.sleep(20)

    return jobs


################
# Analyze output
################


def get_metrics(jobs, regex):
    '''
    Gets CloudWatch metrics for training jobs
    Takes in:
        jobs: A dictionary where training job names are keys.
        regex: a regular expression string to parse the objective metric value.
    Returns a dictionary of training job names as keys and corresponding list
        which contains the objective metric from each log stream.
    '''
    job_metrics = {}
    for job in list(jobs):
        client = boto3.client('logs')
        streams = client.describe_log_streams(logGroupName='/aws/sagemaker/TrainingJobs',
                                              logStreamNamePrefix=job + '/')
        streams = [s['logStreamName'] for s in streams['logStreams']]
        stream_metrics = []
        for stream in streams:
            events = client.get_log_events(logGroupName='/aws/sagemaker/TrainingJobs',
                                           logStreamName=stream)['events']
            message = [e['message'] for e in events]
            metrics = []
            for m in message:
                try:
                    metrics.append(re.search(regex, m).group(1))
                except:
                    pass
            stream_metrics.extend(metrics)
        job_metrics[job] = stream_metrics
    return job_metrics


def table_metrics(jobs, metrics):
    '''
    Returns Pandas DataFrame of jobs, hyperparameter values, and objective metric value
    '''
    job_metrics = jobs.copy()
    for job in list(job_metrics):
        objective = float(metrics[job][-1]) if len(metrics[job]) > 0 else np.nan
        job_metrics[job].update({'objective': objective,
                                 'job_number': int(job.split('-')[-1])})
    return pd.DataFrame.from_dict(job_metrics, orient='index')
