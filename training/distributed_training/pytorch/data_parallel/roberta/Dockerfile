# ARG region

# FROM 763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.9.0-gpu-py38-cu111-ubuntu20.04

# RUN mkdir /dependencies && cd /dependencies && git clone https://github.com/HerringForks/fairseq && cd fairseq && git checkout use_herring && pip install --upgrade pip && pip install --editable ./

# RUN pip install numpy==1.20.1

FROM 570106654206.dkr.ecr.us-west-2.amazonaws.com/diffusion-sagemaker-smddp:smddp-1.4.0-rc0

RUN mkdir /dependencies && cd /dependencies && git clone https://github.com/HerringForks/fairseq && cd fairseq && git checkout use_new_api && pip install --upgrade pip && pip install --editable ./

RUN pip install --upgrade numpy
