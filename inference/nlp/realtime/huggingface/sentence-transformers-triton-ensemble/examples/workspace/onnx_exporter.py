import torch
from transformers import AutoModel
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.onnx")
    parser.add_argument("--model", required=True)
    
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model, torchscript=True)

    bs = 1
    seq_len = 128
    dummy_inputs = (torch.randint(1000, (bs, seq_len),dtype=torch.int), torch.zeros(bs, seq_len, dtype=torch.int))

    torch.onnx.export(
        model,
        dummy_inputs,
        args.save,
        export_params=True,
        opset_version=10,
        input_names=["token_ids", "attn_mask"],
        output_names=["output"],
        dynamic_axes={"token_ids": [0, 1], "attn_mask": [0, 1], "output": [0]},
    )

    print("Saved {}".format(args.save))
