#!/bin/bash

conda create -y -n processing_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=True
conda activate processing_env && conda install -y transformers numpy conda-pack && conda install -y pytorch torchvision torchaudio -c pytorch
conda-pack -n processing_env