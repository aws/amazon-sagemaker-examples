FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    python3.8-dev \
    python3-pip \
    openjdk-11-jdk \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/* 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3.8 get-pip.py


RUN python3.8 -m pip install --no-cache-dir torchserve \
                --no-cache-dir psutil \
                --no-cache-dir captum \
                --no-cache-dir torch \
                --no-cache-dir torchvision \
                --no-cache-dir torchtext \
                --no-cache-dir transformers==4.6.0

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY config.properties /home/model-server/config.properties

EXPOSE 8080 8081

ENV PYTHONIOENCODING=UTF-8
ENTRYPOINT ["bash", "/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

