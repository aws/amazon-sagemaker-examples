import torch
import torchvision.models as models
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()

    resnet50 = models.resnet50(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    resnet50 = resnet50.eval()
    resnet50.to(device)

    resnet50_jit = torch.jit.script(resnet50)
    resnet50_jit.save(args.save)

    print("Saved {}".format(args.save))
