#!/bin/bash

conda create -y -n sd_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
source activate sd_env
export PYTHONNOUSERSITE=True
pip install torch 
pip install transformers ftfy scipy accelerate
pip install diffusers==0.9.0
pip install transformers[onnxruntime]
pip install conda-pack
conda-pack