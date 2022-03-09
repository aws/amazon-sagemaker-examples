ARG region

# Download base PT DLC
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker-v1.1

# Set workspace directory and copy the EfficientNet example repo
WORKDIR /workspace/efficientnet
COPY . .

# Install dependencies in DLC required to run EfficientNet training script
RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN pip install pynvml
RUN pip install nvidia-pyindex
RUN pip install pytorch-quantization

