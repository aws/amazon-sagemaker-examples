FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

RUN apt-get update && apt-get install gcc -y

RUN ${PIP} --no-cache-dir install --upgrade pip

RUN ${PIP} install \
    pandas \
    scikit-learn

WORKDIR /

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN ${PIP} install --no-cache --upgrade \
    sagemaker-pytorch-training