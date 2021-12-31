ARG region

# Download base TF 2.6 DLC
FROM 570106654206.dkr.ecr.${region}.amazonaws.com/diffusion-sagemaker-smddp:smddp-1.4.0-rc0

# Install dependencies in DLC required to run EfficientNet training script
WORKDIR /workspace/efficientnet

COPY . .

RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN pip install pynvml

# smddp: dali dataloader
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

