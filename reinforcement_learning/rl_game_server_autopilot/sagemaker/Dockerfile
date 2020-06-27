ARG CPU_OR_GPU
ARG AWS_REGION
FROM 520713654638.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-tensorflow:ray0.5.3-${CPU_OR_GPU}-py3

WORKDIR /opt/ml

############################################
# Update Ray version
############################################

RUN pip install --upgrade \
    pip \
    setuptools
    
RUN pip install sagemaker-containers --upgrade

RUN pip install -U ray[rllib]

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train; import ray; from sagemaker_containers.cli.train import main"

# Make things a bit easier to debug
WORKDIR /opt/ml/code
