#!/bin/bash
MODEL_NAME=$1
python -m pip install transformers==4.9.1
python onnx_exporter.py --model $MODEL_NAME

trtexec \
    --onnx=model.onnx \
    --saveEngine=model.plan \
    --minShapes=token_ids:1x128,attn_mask:1x128 \
    --optShapes=token_ids:16x128,attn_mask:16x128 \
    --maxShapes=token_ids:32x128,attn_mask:32x128 \
    --verbose \
    --workspace=14000 \
| tee conversion.txt