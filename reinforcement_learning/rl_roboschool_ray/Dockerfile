ARG CPU_OR_GPU
ARG AWS_REGION
FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.2-tf-${CPU_OR_GPU}-py36

WORKDIR /opt/ml

# Roboschool dependencies

RUN apt-get update && apt-get install -y \
      git cmake ffmpeg pkg-config \
      qtbase5-dev libqt5opengl5-dev libassimp-dev \
      libtinyxml-dev \
      libgl1-mesa-dev \
    && cd /opt \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libboost-python-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.6-dev \
    && ln -s -f /usr/bin/python3.6 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade \
    pip \
    setuptools \
    setproctitle \
    lz4 \
    psutil

RUN pip install sagemaker-containers --upgrade

RUN pip install roboschool==1.0.46

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train;import roboschool; import ray; from sagemaker_containers.cli.train import main"

# Make things a bit easier to debug
WORKDIR /opt/ml/code
