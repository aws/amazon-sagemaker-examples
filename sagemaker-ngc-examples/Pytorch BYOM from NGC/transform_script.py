
import torch
import os

def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, 'nvidia_ssdpyt_fp32_190826.pt')
    return model