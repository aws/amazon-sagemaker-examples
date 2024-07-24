#!/bin/bash
python -m pip install transformers==4.9.1
python onnx_exporter.py
python pt_exporter.py
trtexec --onnx=model.onnx --saveEngine=model_bs16.plan --minShapes=token_ids:1x128,attn_mask:1x128 --optShapes=token_ids:16x128,attn_mask:16x128 --maxShapes=token_ids:128x128,attn_mask:128x128 --fp16 --verbose --workspace=14000 | tee conversion_bs16_dy.txt
