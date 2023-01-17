#!/bin/bash

pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

#Accelearating text encoder with TensorRT 
# trtexec --onnx=encoder.onnx --saveEngine=encoder.plan --minShapes=input_ids:1x77 --optShapes=input_ids:4x77 --maxShapes=input_ids:8x77 --fp16
