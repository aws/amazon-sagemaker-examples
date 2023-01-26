"""
Handles generating traffic and creating the ElasticSearch index and dashboard.
"""
import time
import re
import datetime
import random

import requests
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
import numpy as np
from scipy.stats import poisson

from package import config

def generate_metadata():
    """
    Generates medatadata for the HTTP request: a randomized source and a timestamp.
    """
    millisecond_regex = r'\.\d+'
    timestamp = re.sub(millisecond_regex, '', str(datetime.datetime.now()))
    source = random.choice(['Mobile', 'Web', 'Store'])
    result = [timestamp, 'random_id', source]

    return result


def get_data_payload(test_array):
    return {'data':','.join(map(str, test_array)),
            'metadata': generate_metadata()}


def generate_traffic(X_test):
    """
    Using a feature array as input
    """
    while True:
        # NB: The shuffle will mutate the X_test array in-place, so ensure
        # you're working with a copy if you intend to use the calling argument
        # array elsewhere.
        np.random.shuffle(X_test)
        for example in X_test:
            data_payload = get_data_payload(example)
            invoke_endpoint(data_payload)
            # We invoke the function according to a shifted Poisson distribution
            # to simulate data arriving at random intervals
            time.sleep(poisson.rvs(1, size=1)[0])


def invoke_endpoint(payload):
    """
    We get credentials from the IAM role of the notebook instance,
    then use them to create a signed request to the API Gateway
    """
    auth = BotoAWSRequestsAuth(aws_host="{}.execute-api.{}.amazonaws.com".format(
                                 config.REST_API_GATEWAY, config.AWS_REGION),
                               aws_region=config.AWS_REGION,
                               aws_service='execute-api')

    invoke_url = "https://{}.execute-api.{}.amazonaws.com/prod/invocations".format(
        config.REST_API_GATEWAY, config.AWS_REGION)

    requests.post(invoke_url, json=payload, auth=auth)
