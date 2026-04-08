#!/bin/bash
set -e

# ============================================================
# Increase shared memory size for NCCL
# SageMaker default is 64MB which is too small for multi-GPU NCCL
# ============================================================
echo "=== Checking shared memory ==="
df -h /dev/shm
# Try to remount with larger size (may fail if not root, but worth trying)
mount -o remount,size=16G /dev/shm 2>/dev/null || echo "Could not remount /dev/shm (expected in SageMaker)"
df -h /dev/shm

# Fix EFA fork() issue - required for multi-process operations
export FI_EFA_FORK_SAFE=1
export RDMAV_FORK_SAFE=1

# ============================================================
# CRITICAL: NCCL settings for GPUs without NVLink (g6e L40S, g5 A10G)
# Force communication through Shared Memory (not P2P)
# ============================================================
export NCCL_P2P_DISABLE=1       # Disable P2P (broken on g6e/g5)
export NCCL_SHM_DISABLE=0       # ENABLE shared memory (the fallback path!)
export NCCL_NET_GDR_LEVEL=0     # Disable GPUDirect RDMA
export NCCL_NVLS_ENABLE=0       # Disable NVLink SHARP features
export NCCL_IB_DISABLE=1        # Disable InfiniBand
export NCCL_ALGO=Ring           # Force Ring algorithm (works without P2P)
export NCCL_DEBUG=INFO



# Load environment variables from .env.local (baked into image)
if [ -f /opt/ml/code/.env.local ]; then
    echo "=== Loading .env.local ==="
    set -a
    source /opt/ml/code/.env.local
    set +a
fi

echo "=== Environment Variables (NCCL/CUDA) ==="
env | grep -E "^NCCL|^CUDA" | sort

echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "=== Hyperparameters ==="
cat /opt/ml/input/config/hyperparameters.json

echo "=== Starting EasyR1 GRPO training ==="
python -c "
import json
import subprocess
import sys
import os

with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
    hp = json.load(f)

# SageMaker wraps values in quotes, strip them
hp = {k: v.strip('\"') if isinstance(v, str) else v for k, v in hp.items()}

cmd = [sys.executable, '/opt/ml/code/train_script.py']
for k, v in hp.items():
    cmd.extend([f'--{k}', str(v)])

print('Running:', ' '.join(cmd))
# Pass current environment to subprocess (includes all NCCL vars)
subprocess.run(cmd, check=True, env=os.environ.copy())
"
