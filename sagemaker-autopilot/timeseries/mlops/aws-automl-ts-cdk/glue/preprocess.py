#!/usr/bin/env python
# coding: utf-8

import zipfile
import boto3
import pandas as pd
import sys
import os
from awsglue.utils import getResolvedOptions

s3 = boto3.client('s3')


def download_and_extract(bucket, prefix, csv_dir):
    # List objects in the given bucket with the provided prefix
    s3_objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']
    
    # Filter out possible filenames
    possible_filenames = ["data.zip", "tts.csv", "TTS.csv"]
    fileuri = None
    for obj in s3_objects:
        filename = os.path.basename(obj['Key'])
        if filename.lower() in [name.lower() for name in possible_filenames]:
            fileuri = obj['Key']
            break

    # If none of the filenames matched, raise an error
    if not fileuri:
        exit()

    file_name = os.path.join('/tmp', os.path.basename(fileuri))
    print(f"File Name is: {file_name}")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    
    s3.download_file(bucket, fileuri, file_name)

    if fileuri.endswith('.zip'):
        try:
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                zip_ref.extractall(csv_dir)
            return "zip"
        except FileNotFoundError:
            print(f"{file_name} not found.")
            exit()
    elif fileuri.lower().endswith('.csv'):
        source = os.path.join('/tmp', os.path.basename(fileuri))
        destination = os.path.join(csv_dir, 'training_data.csv')
        os.makedirs(csv_dir, exist_ok=True)
        os.rename(source, destination)
        return "csv"
    else:
        print(f"Unsupported file type for {file_name}")
        exit()



def preprocess(csv_dir):
    # Check which files are present
    print("Data merge for ZIP started.")
    tts_present = os.path.exists(os.path.join(csv_dir, 'TTS.csv'))
    rts_present = os.path.exists(os.path.join(csv_dir, 'RTS.csv'))
    metadata_present = os.path.exists(os.path.join(csv_dir, 'metadata.csv'))

    # Load necessary files
    if tts_present:
        tts_df = pd.read_csv(os.path.join(csv_dir, 'TTS.csv'))
    if rts_present:
        rts_df = pd.read_csv(os.path.join(csv_dir, 'RTS.csv'))
    if metadata_present:
        metadata_df = pd.read_csv(os.path.join(csv_dir, 'metadata.csv'))

    # Scenario 1: Only TTS.csv is present
    if tts_present and not rts_present and not metadata_present:
        final_data = tts_df

    # Scenario 2: TTS.csv is present along with one of RTS.csv OR metadata.csv
    elif tts_present and rts_present and not metadata_present:
        final_data = pd.merge(tts_df, rts_df, how='right', on=['product_code', 'location_code', 'timestamp']) # Change the merge columns and type of Merge based on your dataset.
    elif tts_present and not rts_present and metadata_present:
        final_data = pd.merge(tts_df, metadata_df, how='right', on=['product_code']) # Change the merge columns and type of Merge based on your dataset.

    # Scenario 3: All files are present
    elif tts_present and rts_present and metadata_present:
        merged_data = pd.merge(tts_df, rts_df, how='right', on=['product_code', 'location_code', 'timestamp']) # Change the merge columns and type of Merge based on your dataset.
        final_data = pd.merge(merged_data, metadata_df, how='right', on=['product_code']) # Change the merge columns and type of Merge based on your dataset.

    # Error if no recognized pattern is present
    else:
        print("Unrecognized file combination in directory.")
        exit()

    final_data.to_csv(os.path.join(csv_dir, 'training_data.csv'), index=False)
    print(f"Final data merged into: {final_data}")



def save_to_s3(bucket, csv_dir):
    single_csv = 'training_data.csv'
    object_key = os.path.join('input/', os.path.basename(single_csv))
    s3.upload_file(os.path.join(csv_dir, single_csv), bucket, object_key)


# Main Execution
args = getResolvedOptions(sys.argv, ['bucket', 'prefix'])
bucket = args['bucket']
prefix = args['prefix']
csv_dir = 'input/'

file_type = download_and_extract(bucket, prefix, csv_dir)

if file_type == "zip":
    preprocess(csv_dir)

save_to_s3(bucket, csv_dir)
