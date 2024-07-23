#!/usr/bin/env python
import argparse
import boto3
import pandas as pd
import nltk
import sagemaker
import json
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer
from pathlib import Path

def nest_sentences(document):    
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 2000:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = [sentence]
            length = len(sentence)

    if sent:
        nested.append(sent)
        
    final=[]
    for x in nested:
        final.append("".join(x))
        
    return final

def exclude_summary(summary_list):
#     text = summary[0]['summary_text']
    new_summaries=[]
    for summary in summary_list: 
        if "800-273-3217" not in summary: 
            new_summaries.append(summary)
    return new_summaries

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--ticker-cik", type=str, default='amzn')
    parser.add_argument("--endpoint-name", type=str)
    parser.add_argument("--region", type=str)
    args, _ = parser.parse_known_args()
    
    sagemaker_session = sagemaker.Session(boto3.session.Session(region_name=args.region))
    
    #get the csv to data frame data_frame_10k_10q here
    data_frame_10k_10q = pd.read_csv(f'/opt/ml/processing/input/{args.ticker_cik}_10k_10q_recent.csv')
    
    mdna = data_frame_10k_10q.iloc[0]['mdna']
    mdna_chunks = nest_sentences(mdna)
    summary_list = []
    
    print(mdna_chunks)
    
    #initialize predictor from Endpoint
    predictor = sagemaker.predictor.Predictor(endpoint_name=args.endpoint_name, 
                                                sagemaker_session=sagemaker_session,
                                                serializer=JSONSerializer(),
                                                deserializer=JSONDeserializer())    
    # predict for all chunks
    for corpus in mdna_chunks:
        data={}
        data['inputs']=corpus
        summaries = predictor.predict(data)
        summary_list.append(summaries[0]['summary_text'])
        
    
    response = exclude_summary(summary_list)
    
    summary = " ".join(response)
    
    # write the summary output in a JSON File
    result={}
    result['inputs']=summary
    
    out_path = Path(f'/opt/ml/processing/output/{args.ticker_cik}_10k_10q_summary.json')
    out_str = json.dumps(result, indent=4)
    out_path.write_text(out_str, encoding='utf-8')