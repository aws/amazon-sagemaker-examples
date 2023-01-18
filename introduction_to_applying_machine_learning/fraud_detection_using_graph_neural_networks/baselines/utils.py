import os
import pandas as pd

def get_data():
    data_prefix = "preprocessed-data/"

    if not os.path.exists(data_prefix):
        print("""Expected the following folder {} to contain the preprocessed data. 
                 Run data processing first in main notebook before running baselines comparisons""".format(data_prefix))
        return

    features = pd.read_csv(data_prefix + "features_xgboost.csv", header=None)
    labels = pd.read_csv(data_prefix + "tags.csv").set_index('TransactionID')
    valid_users = pd.read_csv(data_prefix + "validation.csv", header=None)
    test_users = pd.read_csv(data_prefix + "test.csv", header=None)
    
    valid_X = features.merge(valid_users, on=[0], how='inner')
    test_X = features.merge(test_users, on=[0], how='inner')
    
    train_index = ~((features[0].isin(test_users[0].values) | (features[0].isin(valid_users[0].values))))   
    train_X = features[train_index]
    valid_y = labels.loc[valid_X[0]]
    test_y = labels.loc[test_X[0]]
    train_y = labels.loc[train_X[0]]
    
    train_X.set_index([0], inplace=True)
    valid_X.set_index([0], inplace=True)
    test_X.set_index([0], inplace=True)

    train_data = train_y.join(train_X)  # first column is the label 'isFraud'
    valid_data = valid_y.join(valid_X)
    test_data = test_y.join(test_X)
    return train_data, valid_data, test_data