ARG BASE_IMAGE="nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3"
FROM ${BASE_IMAGE}

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY resources /opt/program
RUN chmod u+x /opt/program/serve

ENV PATH=/opt/program:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/program
