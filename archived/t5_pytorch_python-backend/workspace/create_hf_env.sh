#!/bin/bash

conda create -y -n hf_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate hf_env
export PYTHONNOUSERSITE=True
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers[sentencepiece]
pip install conda-pack
conda-pack