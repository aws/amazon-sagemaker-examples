import ast
import argparse
import logging
import warnings
import os
import json
import glob
import subprocess
import sys
import boto3
import pickle
import pandas as pd
from collections import Counter
from timeit import default_timer as timer

sys.path.insert(0, 'package')
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from prettytable import PrettyTable
    import autogluon as ag
    from autogluon import TabularPrediction as task
    from autogluon.task.tabular_prediction import TabularDataset


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #

def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(['du','-sh', path]).split()[0].decode('utf-8')

def __load_input_data(path: str) -> TabularDataset:
    """
    Load training data as dataframe
    :param path:
    :return: DataFrame
    """
    input_data_files = os.listdir(path)
    try:
        input_dfs = [pd.read_csv(f'{path}/{data_file}') for data_file in input_data_files]
        return task.Dataset(df=pd.concat(input_dfs))
    except:
        print(f'No csv data in {path}!')
        return None
    
def format_for_print(df):
    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
        table.add_row(row[1:])
    return str(table)

def train(args):
    is_distributed = len(args.hosts) > 1
    host_rank = args.hosts.index(args.current_host)
    dist_ip_addrs = args.hosts
    dist_ip_addrs.pop(host_rank)

    # Load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    train_data = __load_input_data(args.train)
    
    # Extract column info
    target = args.fit_args['label']
    columns = train_data.columns.tolist()
    column_dict = {"columns":columns}
    with open('columns.pkl', 'wb') as f:
        pickle.dump(column_dict, f)
    
    # Train models
    predictor = task.fit(
        train_data=train_data,
        output_directory=args.model_dir,
        **args.fit_args,
    )
    
    # Results summary
    predictor.fit_summary(verbosity=1)

    # Optional test data
    if args.test:
        print(f'Test files: {os.listdir(args.test)}')
        test_data = __load_input_data(args.test)
        # Test data must be labeled for scoring
        if args.fit_args['label'] in test_data:
            # Leaderboard on test data
            print('Running model on test data and getting Leaderboard...')
            leaderboard = predictor.leaderboard(dataset=test_data, silent=True)
            print(format_for_print(leaderboard), end='\n\n')

            # Feature importance on test data
            # Note: Feature importance must be calculated on held-out (test) data.
            # If calculated on training data it will be biased due to overfitting.
            if args.feature_importance:      
                print('Feature importance:')
                # Increase rows to print feature importance                
                pd.set_option('display.max_rows', 500)
                print(predictor.feature_importance(test_data))
        else:
            warnings.warn('Skipping eval on test data since label column is not included.')

    # Files summary
    print(f'Model export summary:')
    print(f"/opt/ml/model/: {os.listdir('/opt/ml/model/')}")
    models_contents = os.listdir('/opt/ml/model/models')
    print(f"/opt/ml/model/models: {models_contents}")
    print(f"/opt/ml/model directory size: {du('/opt/ml/model/')}\n")

# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type','bool', lambda v: v.lower() in ('yes', 'true', 't', '1'))

    # Environment parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    # Arguments to be passed to task.fit()
    parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
                        default="{'presets': ['optimize_for_deployment']}",
                        help='https://autogluon.mxnet.io/api/autogluon.task.html#tabularprediction')
    # Additional options
    parser.add_argument('--feature_importance', type='bool', default=True)

    return parser.parse_args()


if __name__ == "__main__":
    start = timer()
    args = parse_args()
    
    # Verify label is included
    if 'label' not in args.fit_args:
        raise ValueError('"label" is a required parameter of "fit_args"!')

    # Convert optional fit call hyperparameters from strings
    if 'hyperparameters' in args.fit_args:
        for model_type,options in args.fit_args['hyperparameters'].items():
            assert isinstance(options, dict)
            for k,v in options.items():
                args.fit_args['hyperparameters'][model_type][k] = eval(v) 
 
    # Print SageMaker args
    print('fit_args:')
    for k,v in args.fit_args.items():
        print(f'{k},  type: {type(v)},  value: {v}')
        
    # Make test data optional
    if os.environ.get('SM_CHANNEL_TESTING'):
        args.test = os.environ['SM_CHANNEL_TESTING']
    else:
        args.test = None

    train(args)

    # Package inference code with model export
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    subprocess.call('cp columns.pkl /opt/ml/model/code/'.split())

    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {elapsed_time} seconds. Training Completed!')