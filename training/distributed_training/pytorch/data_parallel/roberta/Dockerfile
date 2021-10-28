ARG FROM_IMAGE_NAME=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.9.0-gpu-py38-cu111-ubuntu20.04

FROM ${FROM_IMAGE_NAME}

RUN mkdir /dependencies && cd /dependencies && git clone https://github.com/HerringForks/fairseq && cd fairseq && git checkout use_herring && pip install --upgrade pip && pip install --editable ./

RUN pip install numpy==1.20.1

