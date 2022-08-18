#!/bin/bash

set -e

sudo -u ec2-user -i <<'EOF'

mkdir -p rapids_kernel
cd rapids_kernel
wget -q https://rapidsai-data.s3.us-east-2.amazonaws.com/conda-pack/rapidsai/rapids21.06_cuda11.0_py3.8.tar.gz
echo "wget completed"
tar -xzf *.gz
echo "unzip completed"
source /home/ec2-user/rapids_kernel/bin/activate
conda-unpack 
echo "unpack completed"
python -m ipykernel install --user --name rapids-2106
echo "kernel install completed"

EOF
