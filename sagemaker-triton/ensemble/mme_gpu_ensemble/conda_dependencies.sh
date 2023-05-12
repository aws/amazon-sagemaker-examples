#!/bin/bash

conda create -y -n processing_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=True
conda activate processing_env && conda install torch transformers conda-pack
conda-pack