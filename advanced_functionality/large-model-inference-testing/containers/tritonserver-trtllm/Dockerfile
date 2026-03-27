ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:26.01-trtllm-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://github.com/NVIDIA/TensorRT-LLM.git /opt/TensorRT-LLM
RUN cd /opt/TensorRT-LLM && git fetch origin 751d5f175ca485f26227f69756179d209d8ebbda
RUN cd /opt/TensorRT-LLM && GIT_LFS_SKIP_SMUDGE=1 git reset --hard 751d5f175ca485f26227f69756179d209d8ebbda

RUN pip3 install datasets==3.1.0 evaluate~=0.4.3 rouge_score~=0.1.2 sentencepiece~=0.2.0

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
