FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

RUN pip3 install sagemaker-training

COPY cifar10-training-sagemaker.py /opt/ml/code/cifar10-training-sagemaker.py
COPY model_def.py /opt/ml/code/model_def.py
COPY inference.py /opt/ml/model/code/inference.py
COPY requirements.txt /opt/ml/model/code/requirements.txt

ENV SAGEMAKER_PROGRAM cifar10-training-sagemaker.py
