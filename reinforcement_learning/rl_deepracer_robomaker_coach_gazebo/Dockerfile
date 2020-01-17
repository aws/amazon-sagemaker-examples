#
# This is the custom docker image used in the sagemaker container
#
ARG CPU_OR_GPU
ARG AWS_REGION
FROM 520713654638.dkr.ecr.$AWS_REGION.amazonaws.com/sagemaker-tensorflow-scriptmode:1.12.0-$CPU_OR_GPU-py3

COPY ./src/markov /opt/amazon/markov

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        libav-tools \
        libjpeg-dev \
        libxrender1 \
        python3.6-dev \
        python3-opengl \
        wget \
        xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Redis.
RUN \
    cd /tmp && \
    wget http://download.redis.io/redis-stable.tar.gz && \
    tar xvzf redis-stable.tar.gz && \
    cd redis-stable && \
    make && \
    make install

# Bootstrap the PIP installs to make it faster to re-build the container image on code changes.
RUN pip install \
    annoy==1.8.3 \
    Pillow==4.3.0 \
    matplotlib==2.0.2 \
    numpy==1.17.0 \
    pandas==0.22.0 \
    pygame==1.9.3 \
    PyOpenGL==3.1.0 \
    scipy==1.3.0 \
    scikit-image==0.15.0 \
    futures==3.1.1 \
    boto3==1.9.23 \
    minio==4.0.5 \
    kubernetes==7.0.0 \
    opencv-python==4.1.1.26 \
    rl-coach-slim==1.0.0 \
    retrying \
    eventlet

RUN pip install mxnet-mkl>=1.3.0

RUN pip install --no-cache-dir --upgrade sagemaker-containers

# Patch Intel coach
COPY ./src/rl_coach.patch /opt/amazon/rl_coach.patch
RUN patch -p1 -N --directory=/usr/local/lib/python3.6/dist-packages/ < /opt/amazon/rl_coach.patch



# Copy in all the code and make it available on the path
COPY ./src/lib/redis.conf /etc/redis/redis.conf
ENV PYTHONPATH /opt/amazon/:$PYTHONPATH
ENV PATH /opt/ml/code/:$PATH
WORKDIR /opt/ml/code

# Tell sagemaker-containers where the launch point is for training job.
ENV NODE_TYPE SAGEMAKER_TRAINING_WORKER

ENV PYTHONUNBUFFERED 1