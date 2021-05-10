import argparse
import joblib
import os
import json
import numpy as np
import pandas as pd
import logging
import sys
import pickle
from my_custom_library import cross_validation

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    """Deserialize and return fitted model.
    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    model = pickle.load(open(os.path.join(model_dir, model_file), 'rb'))
    return model


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """
    
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--num_round', type=int, default=os.environ.get('SM_HP_num_round'))
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='reg:squarederror')
    parser.add_argument('--K', type=int, default=os.environ.get('SM_HP_K'))

    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def train():
    """
    Train the PyTorch model
    """

    K = args.K
    
    hyperparameters = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'objective': args.objective,
        'num_round': args.num_round
    }
    
    train_df = pd.read_csv(f'{args.train}/train.csv', header=None)
    
    rmse_list, model = cross_validation(train_df, K, hyperparameters)
    k_fold_avg = sum(rmse_list)/len(rmse_list)
    print(f'RMSE average across folds: {k_fold_avg}')
    
    model_location = args.model_dir + '/xgboost-model'
    pickle.dump(model, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))
    
    
if __name__ == "__main__":
        
    args, _ = parse_args()
    train()
    