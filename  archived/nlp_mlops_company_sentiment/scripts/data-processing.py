#!/usr/bin/env python
import argparse
# from smfinance import SECDataSetConfig, DataLoader
import boto3
import sagemaker
import pandas as pd
import requests
from datetime import date
from dateutil.relativedelta import relativedelta
from smjsindustry.finance import DataLoader
from smjsindustry.finance.processor_config import EDGARDataSetConfig


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--ticker-cik", type=str, default='amzn')
    parser.add_argument("--instance-type", type=str, default="ml.c5.2xlarge")
    parser.add_argument("--region", type=str)
    parser.add_argument("--bucket", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--role", type=str)
       
    args, _ = parser.parse_known_args()
    
    s3_client = boto3.client('s3', region_name=args.region)
    sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=args.region)) 
    
    data_loader = DataLoader(
                    role=args.role,                         # loading job execution role
                    instance_count=1,                       # number of instances to run the loading job, only supports 1 instance for now
                    instance_type=args.instance_type,       # instance type to run the loading job
                    volume_size_in_gb=30,                   # size in GB of the EBS volume to use
                    volume_kms_key=None,                    # KMS key for the processing volume
                    output_kms_key=None,                    # KMS key ID for processing job outputs
                    max_runtime_in_seconds=None,            # timeout in seconds. Default is 24 hours.
                    sagemaker_session=sagemaker_session,    # session object
                    tags=None)                              # a list of key-value pairs

    from_date = date.today() + relativedelta(months=-6)
    to_date = date.today()
    
    dataset_config = EDGARDataSetConfig(
                        tickers_or_ciks=[args.ticker_cik],                               # Also supports CIK. Multiple tickers or CIKs can be listed      
                        form_types=['10-K', '10-Q'],                                # list of SEC form types
                        filing_date_start=from_date.strftime("%Y-%m-%d"),                                       # starting filing date
                        filing_date_end=to_date.strftime("%Y-%m-%d"),                                         # ending filing date
                        email_as_user_agent='test-user@test.com')
        
    data_loader.load(
        dataset_config,
        's3://{}/{}'.format(args.bucket, f'{args.prefix}/{args.ticker_cik}'),  # output s3 prefix (both bucket and folder names are required)
        f'{args.ticker_cik}_10k_10q.csv',                         # output file name
        wait=True,
        logs=True)
    
    file_name=f'/opt/ml/processing/output/10k10q/{args.ticker_cik}_10k_10q.csv'
    recent_file_name=f'/opt/ml/processing/output/10k10q/{args.ticker_cik}_10k_10q_recent.csv'
    
    #download the file to the container
    s3_client.download_file(args.bucket, '{}/{}'.format(f'{args.prefix}/{args.ticker_cik}', f'{args.ticker_cik}_10k_10q.csv'), 
                            file_name)
    
    #load data into a data frame, get the most recent filing and save it as a csv
    #the *_recent.csv file will be the output of the Pipelines task which will contain a single record
    data_frame_10k_10q = pd.read_csv(file_name).replace('\n',' ', regex=True)
    data_frame_10k_10q.sort_values(by=['filing_date'], ascending=False).iloc[[0]].to_csv(recent_file_name, index=False)
    
    
    #get articles
    from_article_date = date.today() + relativedelta(months=-1)
    to_article_date = date.today()
    
    headers = {'X-Api-Key': '630a6130847544c3aaa73a538ee36579'}
    query = {"q": f'"{args.ticker_cik}"',
             "sources":'bloomberg,fortune,the-wall-street-journal',
             "domains": "bloomberg.com,fortune.com,wsj.com", 
             "from": from_article_date.strftime("%Y-%m-%d"),
             "to": to_article_date.strftime("%Y-%m-%d"), 
             "pageSize": 10,
             "page": 1
            }

    try:        
        response = requests.get("https://newsapi.org/v2/everything", params=query,headers=headers)        
        response.raise_for_status()
        data_frame_articles = pd.json_normalize(response.json(), record_path =['articles'])
        data_frame_articles.to_csv(f'/opt/ml/processing/output/articles/{args.ticker_cik}_articles.csv', index=False)
    except requests.exceptions.TooManyRedirects as error:
        print(error)