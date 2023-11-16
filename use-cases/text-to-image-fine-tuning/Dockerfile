# This Dockerfile has been modified slightly from https://github.com/bmaltais/kohya_ss/blob/v22.1.1/Dockerfile,
# to work with SageMaker training jobs

FROM nvcr.io/nvidia/pytorch:23.04-py3 as base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install -y git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
    python3.10-dev python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Set python 3.10 and cuda 11.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    update-alternatives --set cuda /usr/local/cuda-11.8

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

WORKDIR /app
RUN python3 -m pip install wheel

# Todo: Install torch 2.1.0 for cu121 support (only available as nightly as of writing)
## RUN python3 -m pip install --pre torch ninja setuptools --extra-index-url https://download.pytorch.org/whl/nightly/cu121

# Todo: Install xformers nightly for Torch 2.1.0 support
## RUN python3 -m pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

# Install requirements
COPY ./requirements.txt ./requirements_linux_docker.txt ./
COPY ./setup/docker_setup.py ./setup.py
# Copy the train file to the specified directory that SageMaker training jobs will use
COPY ./train /opt/program/
RUN python3 -m pip install -r ./requirements_linux_docker.txt
RUN python3 -m pip install -r ./requirements.txt

# In early October 2023, PyTorch 2.1 was released, causing a conflict with xFormers that led to
# breaking many repositories that worked previously. The following was the error message:
# WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
#    PyTorch 2.0.1+cu118 with CUDA 1108 (you have 2.1.0+cu121)
#    Python  3.10.11 (you have 3.10.11)
# In order to resolve this issue temporarily, the following line must be added to update the correct dependencies
# In future versions of the Kohya SS git repository, this will likely be updated in the requirements file(s). At that
# point, you may remove this line and use an updated version of the git repo (see the template.yml file to update the version).

# https://github.com/bmaltais/kohya_ss/issues/1651
# https://github.com/facebookresearch/xformers/issues/897
# https://github.com/facebookresearch/xformers/issues/870
# https://github.com/bmaltais/kohya_ss/issues/1588
# https://github.com/bmaltais/kohya_ss/issues/1638
# https://github.com/facebookresearch/xformers/issues/874
RUN python3 -m pip install -v -U xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118

# Replace pillow with pillow-simd
RUN python3 -m pip uninstall -y pillow && \
    CC="cc -mavx2" python3 -m pip install -U --force-reinstall pillow-simd

# Fix missing libnvinfer7
USER root
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

# We do not need this for our purposes
#RUN useradd -m -s /bin/bash appuser && \
#    chown -R appuser: /app && \
#    chown -R appuser: /opt
    
#USER appuser
#COPY --chown=appuser . .

# Copy all the required code from the host to the container, which has the kohya-ss source and dependencies
COPY . .

STOPSIGNAL SIGINT
ENV LD_PRELOAD=libtcmalloc.so
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PATH="$PATH:/home/appuser/.local/bin:/opt/program"

# We will not be using the GUI. Instead, we call this program programatically in the "train" file
CMD python3 "./kohya_gui.py" ${CLI_ARGS} --listen 0.0.0.0 --server_port 7860