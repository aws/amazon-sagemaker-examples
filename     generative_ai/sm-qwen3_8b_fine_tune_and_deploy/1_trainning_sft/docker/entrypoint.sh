#!/bin/bash
set -e

# Fix EFA fork() issue - required for multi-process tokenization
export FI_EFA_FORK_SAFE=1
export RDMAV_FORK_SAFE=1

# Load environment variables from .env.local (baked into image)
if [ -f /opt/ml/code/.env.local ]; then
    echo "=== Loading .env.local ==="
    set -a
    source /opt/ml/code/.env.local
    set +a
fi

echo "=== Hyperparameters ==="
cat /opt/ml/input/config/hyperparameters.json

echo "=== Starting LlamaFactory SFT training ==="
python -c "
import json
import subprocess
import sys

with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
    hp = json.load(f)

# SageMaker wraps values in quotes, strip them
hp = {k: v.strip('\"') if isinstance(v, str) else v for k, v in hp.items()}

cmd = [sys.executable, '/opt/ml/code/train_script.py']
for k, v in hp.items():
    cmd.extend([f'--{k}', str(v)])

print('Running:', ' '.join(cmd))
subprocess.run(cmd, check=True)
"
