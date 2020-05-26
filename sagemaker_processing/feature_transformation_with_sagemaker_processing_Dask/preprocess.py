from __future__ import print_function
from __future__ import unicode_literals
import s3fs
import dask.dataframe as dd
import boto3
import sys
import time
import json
import time
from dask.distributed import Client
import argparse
import os
import warnings
from tornado import gen

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import joblib
import dask.dataframe as dd
s3_client = boto3.resource('s3')



def get_resource_config():
    resource_config_path = '/opt/ml/config/resourceconfig.json'
    with open(resource_config_path, 'r') as f:
        return json.load(f)

def get_ip_from_host(host_name):
    IP_WAIT_TIME = 300
    counter = 0
    ip = ''

    while counter < IP_WAIT_TIME and ip == '':
        try:
            ip = socket.gethostbyname(host_name)
            break
        except:
            counter += 1
            time.sleep(1)

    if counter == IP_WAIT_TIME and ip == '':
        raise Exception("Exceeded max wait time of 300s for hostname resolution")

    return ip



def upload_objects(bucket, prefix, local_path):
    try:
        bucket_name = bucket #s3 bucket name
        root_path = local_path # local folder for upload

        s3_bucket = s3_client.Bucket(bucket_name)

        for path, subdirs, files in os.walk(root_path):
            for file in files:
                s3_bucket.upload_file(os.path.join(path, file), prefix +'/output/'+file)

    except Exception as err:
        print(err)

async def stop(dask_scheduler):
    await gen.sleep(0.1)
    await dask_scheduler.close()
    loop = dask_scheduler.loop
    loop.add_callback(loop.stop)


def print_shape(df):
    negative_examples, positive_examples = np.bincount(df['income'])
    print('Data shape: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    #Get processor scrip arguments
    args_iter = iter(sys.argv[1:])
    script_args = dict(zip(args_iter, args_iter))
    scheduler_ip = sys.argv[-1]
    
    
    #Start the Dask cluster client
    client = Client('tcp://' + str(scheduler_ip) + ':8786')
    print('Printing cluster information: {}'.format(client))
    
    columns = ['age', 'education', 'major industry code', 'class of worker', 'num persons worked for employer','capital gains', 'capital losses', 'dividends from stocks', 'income']
    class_labels = [' - 50000.', ' 50000+.']
    input_data_path = 's3://' + os.path.join(script_args['s3_input_bucket'], script_args['s3_input_key_prefix'],'census-income.csv')
    print(input_data_path)
    
    #Creating the necessary paths to save the output files
    if not os.path.exists('/opt/ml/processing/train'):
        os.makedirs('/opt/ml/processing/train')
        os.makedirs('/opt/ml/processing/test')
    
    print('Reading input data from {}'.format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df, columns=columns)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.replace(class_labels, [0, 1], inplace=True)
    
    negative_examples, positive_examples = np.bincount(df['income'])
    print('Data after cleaning: {}, {} positive examples, {} negative examples'.format(df.shape, positive_examples, negative_examples))
    
    split_ratio = args.train_test_split_ratio
    print('Splitting data into train and test sets with ratio {}'.format(split_ratio))
    X_train, X_test, y_train, y_test = train_test_split(df.drop('income', axis=1), df['income'], test_size=split_ratio, random_state=0)

    preprocess = make_column_transformer(
        (KBinsDiscretizer(encode='onehot-dense', n_bins=4), ['age', 'num persons worked for employer']),
        (StandardScaler(), ['capital gains', 'capital losses', 'dividends from stocks']),
        (OneHotEncoder(sparse=False), ['education', 'major industry code', 'class of worker'])
    )
    
    
    
    
    print('Running preprocessing and feature engineering transformations in Dask')
    with joblib.parallel_backend('dask'):
        train_features = preprocess.fit_transform(X_train)
        test_features = preprocess.transform(X_test)
    
    print('Train data shape after preprocessing: {}'.format(train_features.shape))
    print('Test data shape after preprocessing: {}'.format(test_features.shape))
    
    train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
    train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
    
    test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
    test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')
    
    print('Saving training features to {}'.format(train_features_output_path))
    pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
    
    print('Saving test features to {}'.format(test_features_output_path))
    pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)
    
    print('Saving training labels to {}'.format(train_labels_output_path))
    y_train.to_csv(train_labels_output_path, header=False, index=False)
    
    print('Saving test labels to {}'.format(test_labels_output_path))
    y_test.to_csv(test_labels_output_path, header=False, index=False)
    upload_objects(script_args['s3_output_bucket'], script_args['s3_output_key_prefix'],'/opt/ml/processing/train/')
    upload_objects(script_args['s3_output_bucket'], script_args['s3_output_key_prefix'],'/opt/ml/processing/test/')
    
    #Calculate the processed dataset baseline statistics on the Dask cluster
    dask_df = dd.read_csv(train_features_output_path)
    dask_df = client.persist(dask_df)
    baseline = dask_df.describe().compute()
    print(baseline)
    print(type(baseline))
    
    #shutdown workers and gracefully exist the script
    client.run_on_scheduler(stop, wait=False)
    sys.exit(os.EX_OK)
