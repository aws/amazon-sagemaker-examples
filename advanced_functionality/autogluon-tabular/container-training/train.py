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
    warnings.filterwarnings("ignore",category=DeprecationWarning)
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

def train(args):

    is_distributed = len(args.hosts) > 1
    host_rank = args.hosts.index(args.current_host)    
    dist_ip_addrs = args.hosts
    dist_ip_addrs.pop(host_rank)
    ngpus_per_trial = 1 if args.num_gpus > 0 else 0

    # load training and validation data
    print(f'Train files: {os.listdir(args.train)}')
    train_data = __load_input_data(args.train)
    print(f'Label counts: {dict(Counter(train_data[args.label]))}')
    print(f'hp: {args.hyperparameters}')
    predictor = task.fit(
        train_data=train_data,
        label=args.label,            
        output_directory=args.model_dir,
        problem_type=args.problem_type,
        eval_metric=args.eval_metric,
        stopping_metric=args.stopping_metric,
        auto_stack=args.auto_stack, # default: False
        hyperparameter_tune=args.hyperparameter_tune, # default: False
        feature_prune=args.feature_prune, # default: False
        holdout_frac=args.holdout_frac, # default: None
        num_bagging_folds=args.num_bagging_folds, # default: 0
        num_bagging_sets=args.num_bagging_sets, # default: None
        stack_ensemble_levels=args.stack_ensemble_levels, # default: 0
        hyperparameters=args.hyperparameters,
        cache_data=args.cache_data,
        time_limits=args.time_limits,
        num_trials=args.num_trials, # default: None
        search_strategy=args.search_strategy, # default: 'random'
        search_options=args.search_options,
        visualizer=args.visualizer,
        verbosity=args.verbosity
    )
    
    # Results summary
    predictor.fit_summary(verbosity=1)

    # Leaderboard on optional test data
    if args.test:
        print(f'Test files: {os.listdir(args.test)}')
        test_data = __load_input_data(args.test)    
        print('Running model on test data and getting Leaderboard...')
        leaderboard = predictor.leaderboard(dataset=test_data, silent=True)
        def format_for_print(df):
            table = PrettyTable(list(df.columns))
            for row in df.itertuples():
                table.add_row(row[1:])
            return str(table)
        print(format_for_print(leaderboard), end='\n\n')

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
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1')
    parser.register('type','bool',str2bool) # add type keyword to registries

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))    
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR']) # /opt/ml/model
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default='') # /opt/ml/input/data/test
    parser.add_argument('--label', type=str, default='label',
                        help="Name of the column that contains the target variable to predict.")
    
    parser.add_argument('--problem_type', type=str, default=None,
                        help=("Type of prediction problem, i.e. is this a binary/multiclass classification or "
                              "regression problem options: 'binary', 'multiclass', 'regression'). "
                              "If `problem_type = None`, the prediction problem type is inferred based "
                              "on the label-values in provided dataset."))
    parser.add_argument('--eval_metric', type=str, default=None,
                        help=("Metric by which predictions will be ultimately evaluated on test data."
                              "AutoGluon tunes factors such as hyperparameters, early-stopping, ensemble-weights, etc. "
                              "in order to improve this metric on validation data. "
                              "If `eval_metric = None`, it is automatically chosen based on `problem_type`. "
                              "Defaults to 'accuracy' for binary and multiclass classification and "
                              "'root_mean_squared_error' for regression. "
                              "Otherwise, options for classification: [ "
                              "    'accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted', "
                              "    'roc_auc', 'average_precision', 'precision', 'precision_macro', 'precision_micro', 'precision_weighted', "
                              "    'recall', 'recall_macro', 'recall_micro', 'recall_weighted', 'log_loss', 'pac_score']. "
                              "Options for regression: ['root_mean_squared_error', 'mean_squared_error', "
                              "'mean_absolute_error', 'median_absolute_error', 'r2']. "
                              "For more information on these options, see `sklearn.metrics`: "
                              "https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics "
                              "You can also pass your own evaluation function here as long as it follows formatting of the functions "
                              "defined in `autogluon/utils/tabular/metrics/`. "))
    parser.add_argument('--stopping_metric', type=str, default=None,
                        help=("Metric which models use to early stop to avoid overfitting. "
                              "`stopping_metric` is not used by weighted ensembles, instead weighted ensembles maximize `eval_metric`. "
                              "Defaults to `eval_metric` value except when `eval_metric='roc_auc'`, where it defaults to `log_loss`."))      
    parser.add_argument('--auto_stack', type='bool', default=False,
                        help=("Whether to have AutoGluon automatically attempt to select optimal "
                              "num_bagging_folds and stack_ensemble_levels based on data properties. "
                              "Note: Overrides num_bagging_folds and stack_ensemble_levels values. "
                              "Note: This can increase training time by up to 20x, but can produce much better results. "
                              "Note: This can increase inference time by up to 20x."))
    parser.add_argument('--hyperparameter_tune', type='bool', default=False,
                        help=("Whether to tune hyperparameters or just use fixed hyperparameter values "
                              "for each model. Setting as True will increase `fit()` runtimes."))
    parser.add_argument('--feature_prune', type='bool', default=False,
                        help="Whether or not to perform feature selection.")
    parser.add_argument('--holdout_frac', type=float, default=None, 
                        help=("Fraction of train_data to holdout as tuning data for optimizing hyperparameters "
                              "(ignored unless `tuning_data = None`, ignored if `num_bagging_folds != 0`). "
                              "Default value is selected based on the number of rows in the training data. "
                              "Default values range from 0.2 at 2,500 rows to 0.01 at 250,000 rows. "
                              "Default value is doubled if `hyperparameter_tune = True`, up to a maximum of 0.2. "
                              "Disabled if `num_bagging_folds >= 2`."))    
    parser.add_argument('--num_bagging_folds', type=int, default=0, 
                        help=("Number of folds used for bagging of models. When `num_bagging_folds = k`, "
                              "training time is roughly increased by a factor of `k` (set = 0 to disable bagging). "
                              "Disabled by default, but we recommend values between 5-10 to maximize predictive performance. "
                              "Increasing num_bagging_folds will result in models with lower bias but that are more prone to overfitting. "
                              "Values > 10 may produce diminishing returns, and can even harm overall results due to overfitting. "
                              "To further improve predictions, avoid increasing num_bagging_folds much beyond 10 "
                              "and instead increase num_bagging_sets. "))    
    parser.add_argument('--num_bagging_sets', type=int, default=None,
                        help=("Number of repeats of kfold bagging to perform (values must be >= 1). "
                              "Total number of models trained during bagging = num_bagging_folds * num_bagging_sets. "
                              "Defaults to 1 if time_limits is not specified, otherwise 20 "
                              "(always disabled if num_bagging_folds is not specified). "
                              "Values greater than 1 will result in superior predictive performance, "
                              "especially on smaller problems and with stacking enabled. "
                              "Increasing num_bagged_sets reduces the bagged aggregated variance without "
                              "increasing the amount each model is overfit."))
    parser.add_argument('--stack_ensemble_levels', type=int, default=0, 
                        help=("Number of stacking levels to use in stack ensemble. "
                              "Roughly increases model training time by factor of `stack_ensemble_levels+1` " 
                              "(set = 0 to disable stack ensembling).  "
                              "Disabled by default, but we recommend values between 1-3 to maximize predictive performance. "
                              "To prevent overfitting, this argument is ignored unless you have also set `num_bagging_folds >= 2`."))
    parser.add_argument('--hyperparameters', type=lambda s: ast.literal_eval(s), default=None,
                        help="Refer to docs: https://autogluon.mxnet.io/api/autogluon.task.html")
    parser.add_argument('--cache_data', type='bool', default=True,
                       help=("Whether the predictor returned by this `fit()` call should be able to be further trained "
                             "via another future `fit()` call. "
                             "When enabled, the training and validation data are saved to disk for future reuse."))
    parser.add_argument('--time_limits', type=int, default=None, 
                        help=("Approximately how long `fit()` should run for (wallclock time in seconds)."
                              "If not specified, `fit()` will run until all models have completed training, "
                              "but will not repeatedly bag models unless `num_bagging_sets` is specified."))
    parser.add_argument('--num_trials', type=int, default=None, 
                        help=("Maximal number of different hyperparameter settings of each "
                              "model type to evaluate during HPO. (only matters if "
                              "hyperparameter_tune = True). If both `time_limits` and "
                              "`num_trials` are specified, `time_limits` takes precedent."))    
    parser.add_argument('--search_strategy', type=str, default='random',
                        help=("Which hyperparameter search algorithm to use. "
                              "Options include: 'random' (random search), 'skopt' "
                              "(SKopt Bayesian optimization), 'grid' (grid search), "
                              "'hyperband' (Hyperband), 'rl' (reinforcement learner)"))      
    parser.add_argument('--search_options', type=lambda s: ast.literal_eval(s), default=None,
                        help="Auxiliary keyword arguments to pass to the searcher that performs hyperparameter optimization.")
    parser.add_argument('--nthreads_per_trial', type=int, default=None,
                        help="How many CPUs to use in each training run of an individual model. This is automatically determined by AutoGluon when left as None (based on available compute).")
    parser.add_argument('--ngpus_per_trial', type=int, default=None,
                        help="How many GPUs to use in each trial (ie. single training run of a model). This is automatically determined by AutoGluon when left as None.")
    parser.add_argument('--dist_ip_addrs', type=list, default=None,
                        help="List of IP addresses corresponding to remote workers, in order to leverage distributed computation.") 
    parser.add_argument('--visualizer', type=str, default='none',
                        help=("How to visualize the neural network training progress during `fit()`. "
                              "Options: ['mxboard', 'tensorboard', 'none']."))          
    parser.add_argument('--verbosity', type=int, default=2, 
                        help=("Verbosity levels range from 0 to 4 and control how much information is printed during fit(). "
                              "Higher levels correspond to more detailed print statements (you can set verbosity = 0 to suppress warnings). "
                              "If using logging, you can alternatively control amount of information printed via `logger.setLevel(L)`, "
                              "where `L` ranges from 0 to 50 (Note: higher values of `L` correspond to fewer print statements, "
                              "opposite of verbosity levels"))
    parser.add_argument('--debug', type='bool', default=False,
                       help=("Whether to set logging level to DEBUG"))                         

    return parser.parse_args()


if __name__ == "__main__":
    start = timer()
    args = parse_args()
    
    # Print SageMaker args
    print('\n====== args ======')
    for k,v in vars(args).items():
        print(f'{k},  type: {type(v)},  value: {v}')
    print()
    
    # Convert AutoGluon hyperparameters from strings
    if args.hyperparameters:
        for model_type,options in args.hyperparameters.items():
            assert isinstance(options, dict)
            for k,v in options.items():
                args.hyperparameters[model_type][k] = eval(v)
        print(f'AutoGluon Hyperparameters: {args.hyperparameters}', end='\n\n')
    
    train(args)

    # Package inference code with model export
    subprocess.call('mkdir /opt/ml/model/code'.split())
    subprocess.call('cp /opt/ml/code/inference.py /opt/ml/model/code/'.split())
    
    elapsed_time = round(timer()-start,3)
    print(f'Elapsed time: {elapsed_time} seconds')  
    print('===== Training Completed =====')
