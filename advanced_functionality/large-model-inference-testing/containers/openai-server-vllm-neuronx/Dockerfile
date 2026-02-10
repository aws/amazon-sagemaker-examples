ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.13.0-neuronx-py312-sdk2.27.1-ubuntu24.04
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PJRT_DEVICE=NEURON \
    LD_LIBRARY_PATH="/opt/conda/lib:/opt/aws/neuron/lib:${LD_LIBRARY_PATH}" \
    PATH="/opt/program:/opt/conda/bin:/opt/aws/neuron/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            ca-certificates \
            build-essential \
            git \
            libssl-dev \
            libcurl4-openssl-dev \
            libgoogle-perftools-dev \
            libnuma-dev \
            pkg-config \
            unzip \
            wget \
            nginx \
      && rm -rf /var/lib/apt/lists/*

# App Setup
COPY resources /opt/program
RUN chmod u+x /opt/program/serve
WORKDIR /opt/program
