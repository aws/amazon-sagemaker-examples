#!/usr/bin/env python

# A sample script for training an ML model
# It does 2 things
# load csv data in /opt/ml/data

from __future__ import print_function

import os
import json
import pickle

import pprint

# print out dictionary in a nice way
pp = pprint.PrettyPrinter(indent=1)


# where SageMaker injects training info inside container
input_dir="/opt/ml/input/"

# SageMaker treat "/opt/ml/model" as checkpoint direcotry
# and it will send everything there to S3 output path you 
# specified 
model_dir="/opt/ml/model"


def main():
    
    print("== Loading hyperparamters ===")
    with open(os.path.join(input_dir, 'config/hyperparameters.json'), 'rb') as f:
        hyp = json.load(f)
    
    print("== Hyperparamters: ==")
    pp.pprint(hyp)
    
    # define your training logic here
    # import tensorflow as pd
    # import pandas as tf

    model = None
    
    print("== Saving model checkpoint ==")
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("== training completed ==")
    return

if __name__ == '__main__':
    main()


