# As part of the transform, we need to pull this image from ECR instead of DockerHub.
# This is a static image of Ubuntu 16.04, maintained by the silverstone-dev team. It is maintained
# in the Alpha AWS account, as that is where BATS builds are configured to take place.
#
# See the following for more information on BATS DockerImage: https://w.amazon.com/index.php/BuilderTools/PackagingTeam/Products/BATS/Transformers/DockerImage
FROM ubuntu:18.04

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
    nginx \
    libssl-dev \
    libffi-dev\
    && rm -rf /var/lib/apt/lists/*

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
    setuptools==39.1.0 \
    annoy==1.8.3 \
    Pillow==4.3.0 \
    matplotlib==2.0.2 \
    numpy==1.14.5 \
    pandas==0.22.0 \
    pygame==1.9.3 \
    PyOpenGL==3.1.0 \
    scipy==1.2.1 \
    scikit-image==0.15.0 \
    futures==3.1.1 \
    boto3==1.9.23 \
    minio==4.0.5 \
    cryptography==3.2.1 \
    kubernetes==7.0.0 \
    opencv-python==4.1.1.26 \
    bokeh==1.4.0 \
    rl-coach-slim==1.0.0 \
    retrying==1.3.3 \
    eventlet==0.26.1 \
    flask==1.1.2 \
    gevent==20.6.2 \
    gunicorn==20.0.4 \
    h5py==2.10.0 \
    pytest==5.4.1 \
    pytest-cov==2.8.1

RUN pip install https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl

# Patch Intel coach
COPY ./src/rl_coach.patch /opt/amazon/rl_coach.patch
RUN patch -p1 -N --directory=/usr/local/lib/python3.6/dist-packages/ < /opt/amazon/rl_coach.patch

# Get the sagemaker-containers library.  At some point it'll be as easy as...
# RUN pip install sagemaker-containers
# But for now we need a custom one so...
COPY ./src/lib/custom-sagemaker-containers.sh /tmp/custom-sagemaker-containers.sh
RUN /tmp/custom-sagemaker-containers.sh
# This (SAGEMAKER_TRAINING_MODULE bootstrap) will go away with future version of sagemaker-containers
ENV SAGEMAKER_TRAINING_MODULE sagemaker_bootstrap:train

# Copy in all the code and make it available on the path
COPY ./src/lib/model_validator /opt/ml/code/model_validator
COPY ./src/lib/sample_data /opt/ml/code/sample_data
COPY ./src/lib/serve /opt/ml/code/serve
COPY ./src/lib/nginx.conf /opt/ml/code/nginx.conf
COPY ./src/lib/sagemaker_bootstrap.py /opt/ml/code/sagemaker_bootstrap.py
COPY ./src/lib/sage-train.sh /opt/ml/code/sage-train.sh
COPY ./src/lib/redis.conf /etc/redis/redis.conf

ENV PYTHONPATH /opt/ml/code/:/opt/amazon/:$PYTHONPATH
ENV PATH /opt/ml/code/:$PATH
WORKDIR /opt/ml/code

# Tell sagemaker-containers where the launch point is for training job.
ENV SAGEMAKER_TRAINING_COMMAND /opt/ml/code/sage-train.sh
ENV NODE_TYPE SAGEMAKER_TRAINING_WORKER

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1