
import torch
import os

def model_fn(model_dir):
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32',map_location='cpu')
    return model