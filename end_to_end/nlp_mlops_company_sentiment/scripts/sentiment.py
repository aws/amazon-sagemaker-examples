#!/usr/bin/env python
import argparse
import boto3
import pandas as pd
import sagemaker
import json
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from botocore.exceptions import ClientError
import logging
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    # parameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--ticker-cik", type=str, default='amzn')
    parser.add_argument("--endpoint-name", type=str)
    parser.add_argument("--region", type=str)
    args, _ = parser.parse_known_args()
    
    sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=args.region))
    
    #get the json data 
    f = open(f'/opt/ml/processing/input/10k10q/{args.ticker_cik}_10k_10q_summary.json',)  
    # returns JSON object as 
    # a dictionary
    sec_summary = json.load(f)
    sec_summary['inputs'] = sec_summary['inputs'][:2500]
    sec_summary['source'] = f'{args.ticker_cik} SEC Report'
    sec_df = pd.json_normalize(sec_summary)
    sec_df = sec_df[['source', 'inputs']]
    
    articles_df = pd.read_csv(f'/opt/ml/processing/input/articles/{args.ticker_cik}_articles.csv')
    articles_df = articles_df[['source.name', 'content', 'description']]
    articles_df['inputs'] = articles_df[['content', 'description']].apply(lambda x: ''.join(x), axis=1)
    articles_df.drop(['content', 'description'], axis=1, inplace=True)
    articles_df.rename(columns={'source.name': 'source'}, inplace=True)
    
    df = sec_df.append(articles_df,ignore_index=True)
    
    data={}
    data['inputs'] = df['inputs'].tolist()
    
    #initialize predictor from Endpoint
    predictor = sagemaker.predictor.Predictor(endpoint_name=args.endpoint_name, 
                                                sagemaker_session=sagemaker_session,
                                                serializer=JSONSerializer(),
                                                deserializer=JSONDeserializer())    
    # predict for all chunks
    try:
        response = predictor.predict(data)
        response_df = pd.json_normalize(response)
        response_df['source'] = df['source']
        response_df=response_df[['source', 'label', 'score']]

        response_df.to_csv(f'/opt/ml/processing/output/{args.ticker_cik}_sentiment_result.csv', index=False)
    except ClientError as e:
        stacktrace = traceback.format_exc()
        error_message = e.response["Error"]["Message"]
        LOGGER.error("{}".format(stacktrace))

        raise Exception(error_message)