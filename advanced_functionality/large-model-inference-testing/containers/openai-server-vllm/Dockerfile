ARG BASE_IMAGE="nvcr.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04"
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python 3 and pip3
RUN apt-get update && apt-get install -y \
    wget \
    nginx \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Remove the externally-managed marker
RUN rm -f /usr/lib/python*/EXTERNALLY-MANAGED

# Install vLLM and its dependencies
RUN pip3 install --no-cache-dir vllm==0.15.0

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
