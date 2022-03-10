ARG region

# Download base PT DLC. Note that this notebook required a DLC with >= PT 1.10.2
FROM 763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker-v1.1

# Set workspace directory and copy the content of the active directory, which is set in the build_and_push.sh script.
WORKDIR /workspace/efficientnet
COPY . .

# Install dependencies in DLC required to run EfficientNet training script
RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN pip install pynvml
RUN pip install nvidia-pyindex
RUN pip install pytorch-quantization

