import xgboost as xgb
import numpy as np


def cross_validation(df, K, hyperparameters):
    """
    Perform cross validation on a dataset.

    :param df: pandas.DataFrame
    :param K: int
    :param hyperparameters: dict
    """
    train_indices = list(df.sample(frac=1).index)
    k_folds = np.array_split(train_indices, K)
    if K == 1:
        K = 2
        
    rmse_list = []
    for i in range(len(k_folds)):
        training_folds = [fold for j, fold in enumerate(k_folds) if j != i]
        training_indices = np.concatenate(training_folds)
        x_train, y_train = df.iloc[training_indices,1:], df.iloc[training_indices,:1]
        x_validation, y_validation = df.iloc[k_folds[i],1:], df.iloc[k_folds[i],:1]
        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dvalidation = xgb.DMatrix(data=x_validation, label=y_validation)

        model = xgb.train(
            params=hyperparameters,
            dtrain=dtrain,
            evals=[(dtrain, 'train'), (dvalidation, 'validation')]
        )
        eval_results = model.eval(dvalidation)
        rmse_list.append(float(eval_results.split('eval-rmse:')[1]))
    return rmse_list, model