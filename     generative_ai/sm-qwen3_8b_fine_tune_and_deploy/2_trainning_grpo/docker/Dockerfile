# EasyR1 GRPO Training Image for SageMaker
# Build from NVIDIA official image instead of pre-built verl image
FROM nvcr.io/nvidia/pytorch:25.05-py3

# ============================================================
# CRITICAL: NCCL settings for GPUs without NVLink (g6e L40S, g5 A10G)
# Set these EARLY before any NCCL initialization
# ============================================================
ENV NCCL_P2P_DISABLE=1
ENV NCCL_SHM_DISABLE=1
ENV NCCL_NET_GDR_LEVEL=0
ENV NCCL_NVLS_ENABLE=0
ENV NCCL_IB_DISABLE=1
ENV NCCL_NET=Socket
ENV NCCL_SOCKET_IFNAME=eth0
ENV NCCL_DEBUG=INFO

# Define environments
ENV MAX_JOBS=32
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PYTHONUNBUFFERED=1

# Fix Triton UnicodeDecodeError when reading ldconfig output
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Workaround for Triton ldconfig UTF-8 issue: pre-set TRITON_LIBCUDA_PATH
ENV TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64

# Install systemctl and tini
RUN apt-get update && \
    apt-get install -y -o Dpkg::Options::="--force-confdef" systemd tini locales && \
    locale-gen en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ldconfig

# Fix Triton ldconfig UnicodeDecodeError by ensuring clean ldconfig cache
RUN rm -f /etc/ld.so.cache && ldconfig

# Upgrade pip
RUN python -m pip install --upgrade pip

# Uninstall nv-pytorch fork to avoid conflicts
RUN pip uninstall -y torch torchvision torchaudio \
    pytorch-quantization pytorch-triton torch-tensorrt \
    transformer-engine flash-attn apex megatron-core \
    xgboost opencv grpcio 2>/dev/null || true

# Remove nv workspace
RUN rm -rf /workspace

# Install torch-2.8.0 + vllm-0.11.0 + dependencies
RUN pip install \
    "vllm==0.11.0" \
    "torch==2.8.0" \
    "torchvision==0.23.0" \
    "torchaudio==2.8.0" \
    tensordict torchdata \
    accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" "grpcio>=1.62.1" "optree>=0.13.0" \
    pandas "ray[default]" codetiming hydra-core pylatexenc \
    qwen-vl-utils wandb liger-kernel

# Install flash-attn-2.8.3
RUN ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp312-cp312-linux_x86_64.whl" && \
    wget -nv -P /tmp "${URL}" && \
    pip install "/tmp/$(basename ${URL})" && \
    rm -f /tmp/*.whl

# Install EasyR1/veRL from GitHub
RUN pip install --upgrade git+https://github.com/hiyouga/EasyR1.git

# Patch Triton to handle non-UTF8 ldconfig output
RUN TRITON_DRIVER=$(python -c "import triton.backends.nvidia.driver as d; print(d.__file__)") && \
    echo "Patching Triton driver at: $TRITON_DRIVER" && \
    sed -i 's/\.decode("utf-8")/.decode("utf-8", errors="replace")/g' "$TRITON_DRIVER" && \
    sed -i 's/\.decode('\''utf-8'\'')/.decode('\''utf-8'\'', errors='\''replace'\'')/g' "$TRITON_DRIVER" && \
    echo "Triton driver patched successfully"

# Install additional dependencies for SageMaker
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy training scripts into image
COPY train_script.py /opt/ml/code/train_script.py
COPY train_config.yaml /opt/ml/code/train_config.yaml
COPY entrypoint.sh /opt/ml/code/entrypoint.sh
COPY .env.local /opt/ml/code/.env.local

# Copy reward function
COPY reward_function/ /opt/ml/code/reward_function/

RUN chmod +x /opt/ml/code/entrypoint.sh

# Write NCCL vars to /etc/profile.d/ for spawned processes
RUN echo 'export NCCL_P2P_DISABLE=1' > /etc/profile.d/nccl.sh && \
    echo 'export NCCL_SHM_DISABLE=1' >> /etc/profile.d/nccl.sh && \
    echo 'export NCCL_NET_GDR_LEVEL=0' >> /etc/profile.d/nccl.sh && \
    echo 'export NCCL_NVLS_ENABLE=0' >> /etc/profile.d/nccl.sh && \
    echo 'export NCCL_IB_DISABLE=1' >> /etc/profile.d/nccl.sh && \
    echo 'export NCCL_NET=Socket' >> /etc/profile.d/nccl.sh && \
    echo 'export NCCL_SOCKET_IFNAME=eth0' >> /etc/profile.d/nccl.sh && \
    chmod +x /etc/profile.d/nccl.sh

# Create SageMaker directories
RUN mkdir -p /opt/ml/model /opt/ml/input /opt/ml/output /workspace/data_parquet

# CRITICAL: Upgrade transformers for Qwen3-VL support
RUN pip install --upgrade "transformers==4.57.0"

# Patch vLLM tokenizer.py - all_special_tokens_extended removed in newer transformers
RUN VLLM_TOKENIZER=$(python -c "from vllm.transformers_utils import tokenizer; print(tokenizer.__file__)") && \
    sed -i 's/\.all_special_tokens_extended/.all_special_tokens/g' "$VLLM_TOKENIZER"

# Patch vLLM qwen3_vl.py - tie_word_embeddings attribute
RUN VLLM_QWEN3VL=$(python -c "import sys; sys.stderr = open('/dev/null', 'w'); from vllm.model_executor.models import qwen3_vl; print(qwen3_vl.__file__)" 2>/dev/null | tail -1) && \
    sed -i 's/if config\.tie_word_embeddings:/if getattr(config, "tie_word_embeddings", False):/g' "$VLLM_QWEN3VL"

# Patch vLLM qwen3.py - tie_word_embeddings attribute
RUN VLLM_QWEN3=$(python -c "import sys; sys.stderr = open('/dev/null', 'w'); import logging; logging.disable(logging.CRITICAL); from vllm.model_executor.models import qwen3; print(qwen3.__file__)" 2>/dev/null | tail -1) && \
    sed -i 's/self\.config\.tie_word_embeddings/getattr(self.config, "tie_word_embeddings", False)/g' "$VLLM_QWEN3"

# Patch verl/fsdp_workers.py - no_init_weights removed in newer transformers
COPY fsdp_workers.py /tmp/fsdp_workers.py
RUN VERL_PATH=$(python -c "import verl; print(verl.__path__[0])") && \
    cp /tmp/fsdp_workers.py "$VERL_PATH/workers/fsdp_workers.py"

# CRITICAL: Patch verl/models/transformers/qwen3_vl.py for transformers >= 4.57.0
# The visual encoder API changed - now returns object with attributes instead of tuple
COPY patch_qwen3_vl.py /tmp/patch_qwen3_vl.py
RUN python3 /tmp/patch_qwen3_vl.py

# Copy model merge script
COPY model_merger.py /opt/ml/code/model_merger.py

WORKDIR /opt/ml/code

# Override base image ENTRYPOINT and set our own
# The NVIDIA base image has an entrypoint that interferes with SageMaker
ENTRYPOINT []
CMD ["/bin/bash", "/opt/ml/code/entrypoint.sh"]
