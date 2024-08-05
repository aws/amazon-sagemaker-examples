import os
os.system('python3 -m pip install -U sagemaker')

import boto3
import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sagemaker.session import Session
from sagemaker.experiments.run import Run, load_run

session = Session(boto3.session.Session(region_name="us-east-1"))


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def read_parameters():
    """
    Read job parameters
    Returns:
        (Namespace): read parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=10)
    parser.add_argument('--cat_feature_cols', type=str, default="sex")
    parser.add_argument('--num_feature_cols', type=str, default="length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight")
    parser.add_argument('--target_col', type=str, default="rings")
    parser.add_argument('--input_path', type=str, default="/opt/ml/processing/input")
    parser.add_argument('--input_filename', type=str, default="abalone.csv")
    parser.add_argument('--output_path_train', type=str, default="/opt/ml/processing/output/train")
    parser.add_argument('--output_path_validation', type=str, default="/opt/ml/processing/output/validation")
    parser.add_argument('--output_path_test', type=str, default="/opt/ml/processing/output/test")
    params, _ = parser.parse_known_args()
    return params


if __name__ == "__main__":
    
    # reading job parameters
    args = read_parameters()

    df = pd.read_csv(
        os.path.join(args.input_path, args.input_filename),
        header=None,
        names=f"{args.cat_feature_cols},{args.num_feature_cols},{args.target_col}".split(',')
    )

    numeric_features = args.num_feature_cols.split(',')
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = args.cat_feature_cols.split(',')
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    y = df.pop("rings")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    train, test = train_test_split(
        X,  
        test_size=args.test_size, 
        random_state=args.random_state
    )

    train, validation  = train_test_split(
        train,
        test_size=args.val_size, 
        random_state=args.random_state
    )

    pd.DataFrame(train).to_csv(
        os.path.join(
            args.output_path_train, 
            "train.csv"
        ), 
        header=False, 
        index=False
    )
    pd.DataFrame(validation).to_csv(
        os.path.join(
            args.output_path_validation, 
            "validation.csv"
        ), 
        header=False, 
        index=False
    )

    pd.DataFrame(test).to_csv(
        os.path.join(
            args.output_path_test, 
            "test.csv"
        ), 
        header=False, 
        index=False
    )

    print("Done")
        
        
