import time
import boto3
import argparse
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import subprocess
import sys
import datetime as datetime

subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument('--feature-group-name-tracks', type=str)
parser.add_argument('--feature-group-name-ratings', type=str)
parser.add_argument('--feature-group-name-user-preferences', type=str)
parser.add_argument('--bucket-name', type=str)
parser.add_argument('--bucket-prefix', type=str)
parser.add_argument('--region', type=str)
args = parser.parse_args()

region = args.region
bucket = args.bucket_name
prefix = args.bucket_prefix
s3_client = boto3.client('s3')
account_id = boto3.client('sts').get_caller_identity()["Account"]
boto_session = boto3.Session(region_name=region)

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region)
sagemaker_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client
)

featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region)

feature_store_session = sagemaker.session.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

feature_group_names = [args.feature_group_name_ratings, args.feature_group_name_tracks, args.feature_group_name_user_preferences]
feature_groups = []
for name in feature_group_names:
    feature_group = FeatureGroup(name=name, sagemaker_session=feature_store_session)
    feature_groups.append(feature_group)


feature_group_s3_prefixes = []
for feature_group in feature_groups:
    feature_group_table_name = feature_group.describe().get("OfflineStoreConfig").get("DataCatalogConfig").get("TableName")
    feature_group_s3_prefix = f'{account_id}/sagemaker/{region}/offline-store/{feature_group_table_name}'
    feature_group_s3_prefixes.append(feature_group_s3_prefix)

# wait for data to be added to offline feature store
def wait_for_offline_store(feature_group_s3_prefix):
    print(feature_group_s3_prefix)
    offline_store_contents = None
    while (offline_store_contents is None):
        objects_in_bucket = s3_client.list_objects(Bucket=bucket, Prefix=feature_group_s3_prefix)
        if ('Contents' in objects_in_bucket and len(objects_in_bucket['Contents']) > 1):
            offline_store_contents = objects_in_bucket['Contents']
        else:
            print('Waiting for data in offline store...')
            time.sleep(60)
    print('Data available:', feature_group_s3_prefix)
    
for s3_prefix in feature_group_s3_prefixes:
    wait_for_offline_store(s3_prefix)
    
    
# once feature groups made available as Offline Feature Store, we query them
tables = { 
    'ratings': {'feature_group': feature_groups[0],
                'cols': ['userid', 'trackid', 'rating']
               },
    'tracks': {'feature_group': feature_groups[1],
                'cols': ['trackid', 'length', 'energy', 'acousticness', 'valence', 'speechiness', 'instrumentalness', 
                        'liveness', 'tempo', 'danceability', 'genre_latin', 'genre_folk', 'genre_blues', 'genre_rap', 
                        'genre_reggae', 'genre_jazz', 'genre_rnb', 'genre_country', 'genre_electronic', 'genre_pop_rock']
               },
    'user_5star_features': {'feature_group': feature_groups[2],
                            'cols': ['userid', 'energy_5star', 'acousticness_5star', 'valence_5star','speechiness_5star', 
                                     'instrumentalness_5star',  'liveness_5star','tempo_5star', 'danceability_5star', 
                                     'genre_latin_5star', 'genre_folk_5star', 'genre_blues_5star', 'genre_rap_5star',
                                     'genre_reggae_5star', 'genre_jazz_5star', 'genre_rnb_5star', 'genre_country_5star', 
                                     'genre_electronic_5star', 'genre_pop_rock_5star']
                           }
  }


import datetime as datetime

for k, v in tables.items():
    min1 = datetime.datetime.now()
    print (f"{min1}\n")
    query = v['feature_group'].athena_query()
    print (query)
    joined_cols = ", ".join(v['cols'])

    # limit number of datapoints for training time
    query_string = "SELECT {} FROM \"{}\" LIMIT 500000".format(joined_cols, query.table_name)
    print(query_string,'\n')

    output_location = f's3://{bucket}/{prefix}/query_results/'
    query.run(query_string=query_string, output_location=output_location)
    query.wait()

    tables[k]['df'] = query.as_dataframe()

    min2 = datetime.datetime.now()
    diff=min2-min1
    diffs = diff.total_seconds()
    print (f"{query_string} --> {diffs} \n")

    
#===========================================

print('Data retrieved from feature store')

ratings = tables['ratings']['df']
tracks = tables['tracks']['df']
user_prefs = tables['user_5star_features']['df']

print('Merging datasets...')
print(f'Ratings: {ratings.shape}\nTracks: {tracks.shape}\nUser Prefs: {user_prefs.shape}\n')

dataset = pd.merge(ratings, tracks, on='trackid', how='inner')
dataset = pd.merge(dataset, user_prefs, on='userid', how='inner')
dataset.drop_duplicates(inplace=True)
dataset.drop(['userid', 'trackid'], axis=1, inplace=True)

# split data
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.2, random_state=42)
print("Training dataset shape: {}\nValidation dataset shape: {}\n".format(train.shape, test.shape))

# Write train, test splits to output path
train_output_path = pathlib.Path('/opt/ml/processing/output/train')
test_output_path = pathlib.Path('/opt/ml/processing/output/test')
train.to_csv(train_output_path / 'train.csv', header=False, index=False)
test.to_csv(test_output_path / 'test.csv', header=False, index=False)

print('Training and Testing Sets Created')