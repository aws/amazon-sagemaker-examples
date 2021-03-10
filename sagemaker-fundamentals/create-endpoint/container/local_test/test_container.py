# This script tests the your own container before running
# on SageMaker infrastructure. It mimics how SageMaker provides
# training info to your container and how it executes it. 

import docker
import os

dirname = os.path.dirname(
    os.path.realpath(__file__)
    )

client = docker.from_env()

try:
    container = client.containers.run(
        'example-serve:latest', 'serve',  
        volumes={
            # mount ml/ to /opt/ml as volume
            # it's a mechanism for the operating 
            # system to share files with the container
            os.path.join(dirname, 'ml') : {'bind': '/opt/ml', 'mode': 'rw'}, 
            },
        ports={
            "8080/tcp": 8080 # expose port 8080 inside container as port 8080 on the host
            },
        remove=True, 
        stderr=True,
        detach=True,
        )
    container.wait()
    byte_str = container.logs()
    print(byte_str.decode('uft-8'))
except KeyboardInterrupt:
    print("stopping the container")
    container.stop()

