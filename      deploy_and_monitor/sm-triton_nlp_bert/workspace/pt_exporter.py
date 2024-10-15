import torch
from transformers import BertModel
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()

    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

    bs = 128
    seq_len = 128
    dummy_inputs = [
        torch.randint(1000, (bs, seq_len)).to(device),
        torch.zeros(bs, seq_len, dtype=torch.int).to(device),
    ]
    model = model.eval()
    model.to(device)

    traced_model = torch.jit.trace(model, dummy_inputs)
    torch.jit.save(traced_model, args.save)

    print("Saved {}".format(args.save))
