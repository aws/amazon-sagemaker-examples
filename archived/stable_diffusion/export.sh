#!/bin/bash

pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

