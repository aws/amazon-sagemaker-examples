ARG BASE_IMAGE

FROM $BASE_IMAGE

RUN git clone -b naman-inf2-example-refactor https://github.com/namannandan/serve.git \
    && cd serve
WORKDIR /serve
RUN pip install pygit2
RUN python ts_scripts/install_from_src.py
ENV LD_LIBRARY_PATH /opt/aws/neuron/lib/:/lib/x86_64-linux-gnu:/opt/conda/lib/:$LD_LIBRARY_PATH
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -
RUN apt-get update -y
RUN apt-get install linux-headers-$(uname -r) -y
RUN apt-get install aws-neuronx-collectives=2.15.* -
