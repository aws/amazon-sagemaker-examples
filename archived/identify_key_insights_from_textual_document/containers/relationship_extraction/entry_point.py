# flake8: noqa: F401
import os
import sys

# import deployment functions
from package.inference import input_fn
from package.inference import model_fn
from package.inference import output_fn
from package.inference import predict_fn
# import training function
from package.training import parse_args
from package.training import train_fn


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
