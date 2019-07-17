# As part of the transform, we need to pull this image from ECR instead of DockerHub. 
# This is a static image of Ubuntu 16.04, maintained by the silverstone-dev team. It is maintained 
# in the Alpha AWS account, as that is where BATS builds are configured to take place.
# 
# See the following for more information on BATS DockerImage:
# https://w.amazon.com/index.php/BuilderTools/PackagingTeam/Products/BATS/Transformers/DockerImage
#

FROM ubuntu:16.04

COPY ./src/markov /opt/amazon/markov

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    wget \
    fonts-liberation \
    libxss1 libappindicator1 libindicator7 \
    xvfb \
    libasound2 \
    libnspr4 \
    libnss3 \
    lsb-release \
    zip \
    xdg-utils \
    libpng-dev \
    python3 \
    python3-pip \
    libssl-dev \
    libffi-dev

# Install Redis.
RUN \
    cd /tmp && \
    wget http://download.redis.io/redis-stable.tar.gz && \
    tar xvzf redis-stable.tar.gz && \
    cd redis-stable && \
    make && \
    make install

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Bootstrap the PIP installs to make it faster to re-build the container image on code changes.
RUN pip install \
    annoy==1.8.3 \
    Pillow==4.3.0 \
    matplotlib==2.0.2 \
    numpy==1.14.0 \
    pandas==0.20.2 \
    pygame==1.9.3 \
    PyOpenGL==3.1.0 \
    scipy==1.2.1 \
    scikit-image==0.13.0 \
    gym==0.10.5 \
    rl-coach-slim==0.11.1 \
    retrying \
    eventlet \
    boto3 \
    minio==4.0.5

RUN pip install kubernetes

RUN pip install https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp35-cp35m-linux_x86_64.whl

RUN pip install mxnet-mkl>=1.3.0

# Patch intel coach so that discrete distributions can not produce nan's
COPY src/lib/ppo_head.py /usr/local/lib/python3.5/dist-packages/rl_coach/architectures/tensorflow_components/heads/ppo_head.py

# Get the sagemaker-containers library.  At some point it'll be as easy as...
RUN pip install sagemaker-containers

# Copy in all the code and make it available on the path
COPY src/lib/redis.conf /etc/redis/redis.conf
ENV PYTHONPATH /opt/amazon/:$PYTHONPATH
ENV PATH /opt/ml/code/:$PATH
WORKDIR /opt/ml/code

# Tell sagemaker-containers where the launch point is for training job.
ENV SAGEMAKER_TRAINING_COMMAND /opt/ml/code/sage-train.sh
ENV NODE_TYPE SAGEMAKER_TRAINING_WORKER

ENV PYTHONUNBUFFERED 1