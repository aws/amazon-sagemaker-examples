ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 https://github.com/NVIDIA/TensorRT-LLM.git /opt/TensorRT-LLM
RUN cd /opt/TensorRT-LLM && git fetch origin 42a7b0922fc9e095f173eab9a7efa0bcdceadd0d
RUN cd /opt/TensorRT-LLM && GIT_LFS_SKIP_SMUDGE=1 git reset --hard 42a7b0922fc9e095f173eab9a7efa0bcdceadd0d

RUN pip3 install datasets==3.1.0 evaluate~=0.4.3 rouge_score~=0.1.2 sentencepiece~=0.2.0

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /opt/tensorrtllm_backend
RUN cd /opt/tensorrtllm_backend && git fetch origin ad209ced188cbae19c7041d968dd4e6d3800ece2
RUN cd /opt/tensorrtllm_backend && git reset --hard ad209ced188cbae19c7041d968dd4e6d3800ece2

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
