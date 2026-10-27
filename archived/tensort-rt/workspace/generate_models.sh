#!/bin/bash
python -m pip install transformers==4.9.1
python onnx_exporter.py
trtexec --onnx=model.onnx --saveEngine=model_bs16.plan --minShapes=token_ids:1x512,attn_mask:1x512 --optShapes=token_ids:16x512,attn_mask:16x512 --maxShapes=token_ids:128x512,attn_mask:128x512 --fp16 --verbose --workspace=14000 | tee conversion_bs16_dy.txt