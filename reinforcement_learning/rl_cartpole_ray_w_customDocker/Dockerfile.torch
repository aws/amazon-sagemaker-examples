# ARG AWS_REGION
# ARG CPU_OR_GPU
# ARG SUFFIX
# ARG VERSION
# ARG FRAMEWORK

# check available containers at https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# PyTorch container: 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.7.1-cpu-py36-ubuntu18.04
# TF container v2 CPU: 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.3.1-cpu-py37-ubuntu18.04
# TF container v2 GPU: 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-inference:2.3.1-gpu-py37-cu102-ubuntu18.04
# TF container v1 CPU: 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.5-cpu-py37-ubuntu18.04
# TF container v1 GPU: 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:1.15.5-gpu-py37-cu100-ubuntu18.04
#FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/${FRAMEWORK}-training:${VERSION}-${CPU_OR_GPU}-${SUFFIX}

ARG processor
ARG region
ARG suffix
#FROM 763104351884.dkr.ecr.$region.amazonaws.com/tensorflow-training:2.3.1-$processor-py37-$suffix
FROM 763104351884.dkr.ecr.$region.amazonaws.com/pytorch-training:1.7.1-$processor-py36-$suffix


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        ffmpeg \
        rsync \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    Cython==0.29.21 \
    tabulate \
    tensorboardX \
    gputil \
    gym==0.17.3 \
    lz4 \
    opencv-python-headless \
    PyOpenGL==3.1.0 \
    pyyaml \
    ray==1.1.0 \
    ray[tune]==1.1.0 \
    ray[rllib]==1.1.0 \
    scipy \
    psutil \
    setproctitle \
    dm-tree \
    tensorflow-probability \
    tf_slim \
    sagemaker-tensorflow-training

# https://github.com/ray-project/ray/issues/11773
RUN pip install dataclasses

# https://github.com/aws/sagemaker-rl-container/issues/39
# https://github.com/openai/gym/issues/1785
#RUN pip install pyglet==1.3.2
RUN pip install pyglet==1.4.10

# recording: https://github.com/openai/gym/issues/35
RUN conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge



# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /

COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]


RUN pip install sagemaker-containers --upgrade

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train;import ray; from sagemaker_containers.cli.train import main"

# Make things a bit easier to debug
WORKDIR /opt/ml/code

RUN pip freeze 

RUN python --version

RUN pip freeze | grep tensorflow


    

