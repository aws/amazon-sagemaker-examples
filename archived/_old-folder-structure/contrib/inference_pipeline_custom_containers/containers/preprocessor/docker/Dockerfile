# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

FROM ubuntu:16.04

LABEL maintainer="Amazon AI"

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.6.6

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    libopencv-dev \
    openssh-client \
    openssh-server \
    nginx \
    vim \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Installing Python3
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
    ./configure && make && make install && \
    apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
    make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip

# Upgrading pip and creating symbolic link for python3
RUN ${PIP} --no-cache-dir install --upgrade pip
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN ${PIP} install --no-cache --upgrade \
    numpy==1.16.2 \
    pandas==1.1.2 \
    scikit-learn==0.23.2 \
    requests==2.21.0 \
    scipy==1.2.2 \
    flask==1.1.1 \
    gevent==20.9.0 \
    gunicorn==20.0.4 \
    boto3==1.15.6 \
    sagemaker_containers==2.8.6.post2 \
    joblib==0.16.0

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY code/* /
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

ENTRYPOINT ["python", "main.py"]
