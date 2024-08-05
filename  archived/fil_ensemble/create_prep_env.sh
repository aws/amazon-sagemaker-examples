#!/bin/bash

conda create -y -n preprocessing_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate preprocessing_env
export PYTHONNOUSERSITE=True
conda install -y -c conda-forge pandas scikit-learn
pip install conda-pack
conda-pack
