#!/usr/bin/env python

# A sample script for training an ML model
# It does 2 things
# load csv data in /opt/ml/data

from __future__ import print_function

import os
import pickle

# where SageMaker injects training data inside container
data_dir="/opt/ml/input/data"

# SageMaker treat "/opt/ml/model" as checkpoint direcotry
# and it will send everything there to S3 output path you 
# specified 
model_dir="/opt/ml/model"


# log dir
log_dir="/opt/ml/output"

def main():
    print("== Files in train channel ==")
    for f in os.listdir(os.path.join(data_dir, 'train')):
        print(f)
    
    # define your training logic here
    # import tensorflow as pd
    # import pandas as tf

    model = None

    # validate / test your model
    # using test data
    print("== Files in the test channel ==")
    for f in os.listdir(os.path.join(data_dir, 'test')):
        print(f)
    
    print("== Saving model checkpoint ==")
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    
    print("== training completed ==")


    return

if __name__ == '__main__':
    main()

