ARG BASE_IMAGE=deepjavalibrary/djl-serving:0.32.0-pytorch-inf2
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-dev \
      python3-pip \
      python3-setuptools \
      software-properties-common \
      && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
      pip3 install --upgrade wheel setuptools

RUN apt-get -y install gnupg2
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB > ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg --import  ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB
RUN gpg --no-default-keyring --keyring ./aws_neuron_keyring.gpg  --export >  ./aws_neuron.gpg
RUN mv ./aws_neuron.gpg /etc/apt/trusted.gpg.d/
RUN rm ./GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB

RUN add-apt-repository -y  "deb https://apt.repos.neuron.amazonaws.com jammy main"
RUN apt-get -y update

RUN apt-get -y install aws-neuronx-collectives=2.*
RUN apt-get -y install aws-neuronx-runtime-lib=2.*
RUN apt-get -y install aws-neuronx-tools=2.*

RUN pip3 config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
RUN pip3 install libneuronxla==2.2.12677.0+470fa032 \
      neuronx-cc==2.21.18209.0+043b1bf7 \
      neuronx-distributed==0.15.22404+1f27bddf \
      neuronx-distributed-inference==0.6.10598+a59fdc00 \
      torch-neuronx==2.8.0.2.10.13553+1e4dd6ca \
      transformers-neuronx==0.13.322 \
      optimum-neuron==0.3.0

RUN pip install --upgrade \
    huggingface_hub==0.35.1 \
    diffusers==0.35.1 \
    transformers==4.56.2

ENV PATH=/opt/program:/opt/aws/neuron/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PJRT_DEVICE=NEURON

