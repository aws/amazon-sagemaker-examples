ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.13.0-neuronx-py312-sdk2.27.1-ubuntu24.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=NEURON \
    LD_LIBRARY_PATH="/opt/conda/lib:/opt/aws/neuron/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}" \
    PATH="/opt/program:/opt/aws/neuron/bin:/opt/tritonserver/bin:${PATH}"

# System Build Tools + Modern CMake (Ubuntu 24.04 has CMake 3.28+ in main repos)
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg2 \
    build-essential git nginx pkg-config unzip \
    libssl-dev libcurl4-openssl-dev libgoogle-perftools-dev \
    libnuma-dev libarchive-dev libxml2-dev zlib1g-dev \
    autoconf automake libtool gperf scons patchelf \
    libre2-dev libb64-dev rapidjson-dev \
    libboost-dev cmake cmake-data \
    && rm -rf /var/lib/apt/lists/*

# Python Build Essentials
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel virtualenv build cmake==3.31.10

# Triton Core Build - Configure to use conda Python
RUN git clone --depth=1 --branch=r26.01 https://github.com/triton-inference-server/server.git /server && \
    cd /server && \
    Python3_ROOT_DIR=/opt/conda \
    Python3_EXECUTABLE=/opt/conda/bin/python3 \
    Python3_INCLUDE_DIR=/opt/conda/include/python3.12 \
    Python3_LIBRARY=/opt/conda/lib/libpython3.12.so \
    ./build.py -v --no-container-build --build-dir=/server/build --backend=python \
    --enable-metrics --enable-logging --enable-stats --endpoint="http" --endpoint="grpc" && \
    cp -r /server/build/opt/* /opt/ && \
    cd / && rm -rf /server

# Install VLLM backend for Triton Inference Server
RUN git clone --depth=1 --branch=r26.01 https://github.com/triton-inference-server/vllm_backend.git /tmp/vllm_backend && \
    cd /tmp/vllm_backend && \
    mkdir -p /opt/tritonserver/backends/vllm && \
    cp -r src/* /opt/tritonserver/backends/vllm/ && \
    rm -rf /tmp/vllm_backend

# Application Setup
COPY resources /opt/program
RUN chmod u+x /opt/program/serve
WORKDIR /opt/program
