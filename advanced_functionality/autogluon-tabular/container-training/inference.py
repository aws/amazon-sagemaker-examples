import sys
import os
import argparse
import logging
import warnings
import os
import json
import subprocess

warnings.filterwarnings("ignore",category=FutureWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), '/opt/ml/code/package'))

import pandas as pd
import pickle
from io import StringIO
from timeit import default_timer as timer
from collections import Counter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from prettytable import PrettyTable
    from autogluon import TabularPrediction as task

def make_str_table(df):   
    table = PrettyTable(['index']+list(df.columns))
    for row in df.itertuples():
        table.add_row(row)
    return str(table)

# ------------------------------------------------------------ #
# Hosting methods                                              #
# ------------------------------------------------------------ #

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """
    print(f'Loading model from {model_dir} with contents {os.listdir(model_dir)}')
    net = task.load(model_dir, verbosity=True)    
    return net


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type. ('text/csv')
    :param output_content_type: The (desired) response content type. ('text/csv')
    :return: response payload and content type.
    """
    start = timer()
    
    # text/csv
    if input_content_type == 'text/csv':
        
        # Load dataset
        df = pd.read_csv(StringIO(data))
        ds = task.Dataset(df=df)

        # Predict
        predictions = net.predict(ds)
        print(f'Prediction counts: {Counter(predictions.tolist())}')
        
        # Form response
        output = StringIO()
        pd.DataFrame(predictions).to_csv(output, header=False, index=False)
        response_body = output.getvalue()        
        
        # If target column passed, evaluate predictions performance
        target = net.label_column
        if target in ds:
            print(f'Label column ({target}) found in input data. '
                  'Therefore, evaluating prediction performance...')

            performance = net.evaluate_predictions(y_true=ds[target], y_pred=predictions.tolist(), 
                                                   auxiliary_metrics=True)
            print(json.dumps(performance, indent=4))       
             
    else: 
        raise NotImplementedError("content_type must be 'text/csv'")

    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {round(timer()-start,3)} seconds')           
    
    return response_body, output_content_type
