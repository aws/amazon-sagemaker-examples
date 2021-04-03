# This script tests the your own container before running
# on SageMaker infrastructure. It mimics how SageMaker provides
# training info to your container and how it executes it. 

import docker
import os
import argparse


# get SageMaker session credentials to be injected into the container
parser = argparse.ArgumentParser()
parser.add_argument('--access-key-id', type=str)
parser.add_argument('--secret-access-key', type=str)
parser.add_argument('--session-token', type=str)
args = parser.parse_args()

dirname = os.path.dirname(
    os.path.realpath(__file__)
    )

client = docker.from_env()
container = client.containers.run(
    'example-image:latest', 'train', # docker run example-image:latest train 
    volumes={
        # mount ml/ to /opt/ml as volume
        # it's a mechanism for the operating 
        # system to communicate with inside of
        # a docker container
        os.path.join(dirname, 'ml') : {'bind': '/opt/ml', 'mode': 'rw'}, 
        },
    
    # set environment variables in the container
    environment={
        "AccessKeyId": args.access_key_id,
        "SecretAccessKey": args.secret_access_key,
        "SessionToken": args.session_token
    },
    stderr=True,
    detach=True,
    )

# wait the execution to finish
container.wait()

# retrieve logs
byte_str=container.logs()

# decode byte string to utf-8 encoding
print(byte_str.decode('utf-8'))
