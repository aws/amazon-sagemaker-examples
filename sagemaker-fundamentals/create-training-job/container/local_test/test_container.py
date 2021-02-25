# This script tests the your own container before running
# on SageMaker infrastructure. It mimics how SageMaker runs
# your container. 

# It downloads all objects recursively from an S3 prefix (your data lake)
# save it in ./data

# It mounts ./data to the /opt/ml/data directory in your container as a 
# docker volume;
# It mounts ./checkpoint to the /opt/ml/checkpoint as a docker volume

# Then it runs the container with the command
#   python train.py
# If the container is implemented correctly, then it should read the 
# csv file from ./data and write a `model.pkl` file in ./checkpoint


# Download data from S3 
# import boto3

# mount ./data and ./checkpoint to the container as docker volumes

# Note: this script needs to be run in its directory

import docker
import os

dirname = os.path.dirname(
    os.path.realpath(__file__)
    )

client = docker.from_env()

# sagemaker runs container with 
# docker run <container> train

container = client.containers.run(
    'test:latest', 'train', 
    volumes={
        os.path.join(dirname, 'ml') : {'bind': '/opt/ml', 'mode': 'rw'},
        }
    )


