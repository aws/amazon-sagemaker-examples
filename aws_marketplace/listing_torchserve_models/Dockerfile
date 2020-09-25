FROM ubuntu:latest

ENV PYTHONUNBUFFERED TRUE

RUN apt-get -y update \
    && apt-get -y install --no-install-recommends \
    ca-certificates \
    git \
    python3-dev \
    openjdk-11-jdk \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp \
    && wget https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

RUN pip install --no-cache-dir psutil \
                --no-cache-dir torch \
                --no-cache-dir torchvision
                
RUN git clone https://github.com/pytorch/serve.git \ 
    && pip install ../serve/ \
    && rm -rf /root/.cache

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

RUN mkdir -p /home/model-server/ && mkdir -p /home/model-server/tmp
COPY config.properties /home/model-server/config.properties

WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

