#!/usr/bin/env python

# A sample script for training an ML model
# It does 2 things
# load csv data in /opt/ml/data
# write an empty binary file model.pkl in /opt/ml/checkpoint

from __future__ import print_function

import os
import pickle
import pandas as pd

# where SageMaker injects training data inside container
data_dir="/opt/ml/input/data/train"

# SageMaker treat "/opt/ml/model" as checkpoint direcotry
# and it will send everything there to S3 output path you 
# specified 
model_dir="/opt/ml/model"

def main():
    
    print("== Files in train channel ==")
    for f in os.listdir(data_dir):
        print(f)

    model = None
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("== training completed ==")
    return

if __name__ == '__main__':
    main()

