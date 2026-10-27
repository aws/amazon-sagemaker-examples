FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE

ARG tarball
ARG requirements
ARG setup

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    fakeroot \
    ca-certificates \
    dpkg-dev \
    g++ \
    python3.6-dev \
    python3-pip \
    openjdk-11-jdk \
    curl \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/* 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
RUN cd /tmp \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN python -m pip install --no-cache-dir torchserve
RUN python -m pip install --no-cache-dir torch>=2.6.0

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY config.properties /home/model-server/config.properties


ENV TEMP=/home/model-server/tmp
ADD ${tarball} /home/model-server/code
WORKDIR /home/model-server/code

RUN if [ -f requirements.txt ] && [ ${requirements} = True ]; then python -m pip install --no-cache-dir --force-reinstall -r requirements.txt; fi
RUN if [ -f setup.py ] && [ ${setup} = True ]; then python -m pip install --no-cache-dir --force-reinstall -e code; fi
RUN python -m pip install --force-reinstall git+git://github.com/facebookresearch/dynalab.git

ENV PYTHONIOENCODING=UTF-8
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

