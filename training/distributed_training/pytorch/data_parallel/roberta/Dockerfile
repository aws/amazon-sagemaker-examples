ARG region

FROM 763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN mkdir /dependencies && cd /dependencies && git clone https://github.com/HerringForks/fairseq && cd fairseq && git checkout master_05_18_2022 && pip install --editable ./