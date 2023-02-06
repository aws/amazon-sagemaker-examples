#!/bin/bash

conda create -y -n processing_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate processing_env
export PYTHONNOUSERSITE=True
pip install torch 
pip install transformers
pip install conda-pack
conda-pack