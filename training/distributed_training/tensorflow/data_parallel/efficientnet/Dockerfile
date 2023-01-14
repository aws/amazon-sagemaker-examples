ARG region

# Download base TF 2.6 DLC
FROM 763104351884.dkr.ecr.${region}.amazonaws.com/tensorflow-training:2.6.0-gpu-py38-cu112-ubuntu20.04

# Install dependencies in DLC required to run EfficientNet training script

RUN pip install nvidia-pyindex tensorflow-addons tensorflow-datasets tensorflow-io

RUN pip install git+https://github.com/NVIDIA/dllogger.git

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

RUN pip install pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-tf-plugin-cuda110
