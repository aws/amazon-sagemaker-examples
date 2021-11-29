ARG CPU_OR_GPU
ARG AWS_REGION
ARG RAY_VERSION
ARG SUFFIX
FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-${RAY_VERSION}-tf-${CPU_OR_GPU}-${SUFFIX}

WORKDIR /opt/ml

# Unity dependencies

RUN pip install --upgrade \
    pip \
    gym-unity \
    mlagents-envs

RUN pip install sagemaker-containers --upgrade

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train;import ray; from sagemaker_containers.cli.train import main; from mlagents_envs.environment import UnityEnvironment; from mlagents_envs.registry import default_registry; from gym_unity.envs import UnityToGymWrapper"

# Make things a bit easier to debug
WORKDIR /opt/ml/code
