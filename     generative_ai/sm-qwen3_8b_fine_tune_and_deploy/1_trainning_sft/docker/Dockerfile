# LlamaFactory SFT Training Image for SageMaker
# Base: LlamaFactory official image (includes llamafactory, transformers, peft, etc.)
FROM hiyouga/llamafactory:latest

# Install additional dependencies not in base image
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy training scripts into image
COPY train_script.py /opt/ml/code/train_script.py
COPY train_config.yaml /opt/ml/code/train_config.yaml
COPY entrypoint.sh /opt/ml/code/entrypoint.sh
COPY .env.local /opt/ml/code/.env.local

RUN chmod +x /opt/ml/code/entrypoint.sh

# SageMaker environment
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Create SageMaker directories
RUN mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output

WORKDIR /opt/ml/code

# Override base image ENTRYPOINT and set our own
# The base image has an NVIDIA entrypoint that interferes with SageMaker
ENTRYPOINT []
CMD ["bash", "/opt/ml/code/entrypoint.sh"]
