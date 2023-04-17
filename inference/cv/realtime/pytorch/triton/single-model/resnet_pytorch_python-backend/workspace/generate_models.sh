#!/bin/bash
python onnx_exporter.py
python pt_exporter.py
trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:128x3x224x224 --maxShapes=input:128x3x224x224 --fp16 --verbose | tee conversion.txt
