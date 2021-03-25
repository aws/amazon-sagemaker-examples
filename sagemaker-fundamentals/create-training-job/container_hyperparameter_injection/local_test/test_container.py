# This script tests the your own container before running
# on SageMaker infrastructure. It mimics how SageMaker provides
# training info to your container and how it executes it. 

import docker
import os

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
    stderr=True,
    detach=True,
    )

# wait the execution to finish
container.wait()

# retrieve logs
byte_str=container.logs()

# decode byte string to utf-8 encoding
print(byte_str.decode('utf-8'))
