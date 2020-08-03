from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest, RFE

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

label_column = 'y'
INPUT_FEATURES_SIZE = 100

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(file) for file in input_files ]
    concat_data = pd.concat(raw_data)
    
    number_of_columns_x = concat_data.shape[1]
    y_train = concat_data.iloc[:,number_of_columns_x-1].values
    X_train = concat_data.iloc[:,:number_of_columns_x-1].values
    
    '''Feature selection pipeline'''
    feature_selection_pipe = Pipeline([
                 ('svr', RFE(SVR(kernel="linear"))),# default: eliminate 50%
                 ('f_reg',SelectKBest(f_regression, k=30)),
                ('mut_info',SelectKBest(mutual_info_regression, k=10))
                ])
    
    
    feature_selection_pipe.fit(X_train,y_train)

    joblib.dump(feature_selection_pipe, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
    '''Save selected feature names'''
    feature_names = concat_data.columns[:-1]
    feature_names = feature_names[feature_selection_pipe.named_steps['svr'].get_support()]
    feature_names = feature_names[feature_selection_pipe.named_steps['f_reg'].get_support()]
    feature_names = feature_names[feature_selection_pipe.named_steps['mut_info'].get_support()]
    joblib.dump(feature_names, os.path.join(args.model_dir, "selected_feature_names.joblib"))
    
    print("Selected features are: {}".format(feature_names))
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data))      
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    print("Input data shape at predict_fn: {}".format(input_data.shape))
    if input_data.shape[1] == INPUT_FEATURES_SIZE:
    # This is a unlabelled example, return only the features
        features = model.transform(input_data)
        return features
    
    elif input_data.shape[1] == INPUT_FEATURES_SIZE + 1:
    # Labeled data. Return label and features
        features = model.transform(input_data.iloc[:,:INPUT_FEATURES_SIZE])
        return np.insert(features, 0, input_data[label_column], axis=1)


def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor