ARG region

# Download base PT DLC
FROM 570106654206.dkr.ecr.${region}.amazonaws.com/diffusion-sagemaker-smddp:smddp-1.4.0-rc0

# Set workspace directory and copy the EfficientNet example repo
WORKDIR /workspace/efficientnet
COPY . .

# Install dependencies in DLC required to run EfficientNet training script
RUN pip install git+https://github.com/NVIDIA/dllogger.git
RUN pip install pynvml

# Install DALI dataloader
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

