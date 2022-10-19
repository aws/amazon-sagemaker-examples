#!/bin/bash


# use this script to save ResNet50 model as a .pt file
python pt_exporter.py

# Optional Scripts
# use this script to convert Pytorch model to ONNX format
# python onnx_exporter.py

#use this command to generate a model plan that will be used to host SageMaker Endpoint
#trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:128x3x224x224 --maxShapes=input:128x3x224x224 --fp16 --verbose | tee conversion.txt
