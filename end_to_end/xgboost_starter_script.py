import os
import sys
import pickle
import xgboost as xgb
import argparse
import pandas as pd
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--num_round', type=int, default=999)
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--nfold', type=int, default=5)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    parser.add_argument('--train_data_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))

    args = parser.parse_args()

    data = pd.read_csv(f'{args.train_data_path}/train.csv')
    train = data.drop('fraud', axis=1)
    label = pd.DataFrame(data['fraud'])
    dtrain = xgb.DMatrix(train, label=label)
    
    params = {'max_depth': args.max_depth, 'eta': args.eta, 'objective': args.objective}
    num_boost_round = args.num_round
    nfold = args.nfold
    early_stopping_rounds = args.early_stopping_rounds
    
    cv_results = xgb.cv(
        params = params,
        dtrain = dtrain,
        num_boost_round = num_boost_round,
        nfold = nfold,
        early_stopping_rounds = early_stopping_rounds,
        metrics = ('auc'),
        seed = 0
        )
    
    print(f"[0]#011train-auc:{cv_results.iloc[-1]['train-auc-mean']}")
    print(f"[1]#011validation-auc:{cv_results.iloc[-1]['test-auc-mean']}")
    
    metrics_data = {
        'classification_metrics': {
            'validation:auc': { 'value': cv_results.iloc[-1]['test-auc-mean']},
            'train:auc': {'value': cv_results.iloc[-1]['train-auc-mean']}
        }
    }

    model = xgb.train(params=params, dtrain=dtrain, num_boost_round=len(cv_results))

    # Save the model to the location specified by ``model_dir``
    metrics_location = args.output_data_dir + '/metrics.json'
    model_location = args.model_dir + '/xgboost-model'
    
    
    with open(metrics_location, 'w') as f:
        json.dump(metrics_data, f)
    
    with open(model_location, 'wb') as f:
        pickle.dump(model, f)

