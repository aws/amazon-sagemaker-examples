FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

ENV PYTHONUNBUFFERED TRUE

ARG tarball
ARG requirements
ARG setup

# delete expired nvidia keys and fetch new ones
RUN apt -o Acquire::AllowInsecureRepositories=true \
-o Acquire::AllowDowngradeToInsecureRepositories=true \
update
RUN apt-get install -y wget
RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb 

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
    && curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py \
    && python3 get-pip.py

RUN python -m pip install --no-cache-dir torchserve
RUN python -m pip install --no-cache-dir torch==1.7.1

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY config.properties /home/model-server/config.properties


ENV TEMP=/home/model-server/tmp
ADD ${tarball} /home/model-server/code
WORKDIR /home/model-server/code

RUN if [ -f requirements.txt ] && [ ${requirements} = True ]; then python -m pip install --no-cache-dir --force-reinstall -r requirements.txt; fi
RUN if [ -f setup.py ] && [ ${setup} = True ]; then python -m pip install --no-cache-dir --force-reinstall -e code; fi
RUN python -m pip install --force-reinstall git+https://github.com/facebookresearch/dynalab.git

ENV PYTHONIOENCODING=UTF-8
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

