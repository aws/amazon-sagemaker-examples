from chalice import Chalice
import json
import numpy as np
from time import gmtime,strftime
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import boto3
import sys
import os
import requests

#Global Varibale from model
content_type = "application/json"
accept = "Accept"
min_servers=10
max_servers=100
new_alloc=0
region='us-west-2'

sagemaker_region = 'us-west-2'
sagemaker_client = boto3.client('sagemaker-runtime',region_name=sagemaker_region)
cloudwatch_region = 'us-west-2'
cloudwatch_cli = boto3.client('cloudwatch',region_name=cloudwatch_region)

app = Chalice(app_name='autopilot-server')

#TBD Export to environment variables
endpoint_name ="sagemaker-tensorflow-serving-2019-09-23-20-53-20-237"
namespace = "autopilot"
gs_inventory_url = "https://3vwufqstq5.execute-api.us-west-2.amazonaws.com/api/currsine24h"


supported_calls={"SupportedCalls":
                  {
                     "/predict":"get game-server predictions",
                     "/currtime":"get the current time in the cloud",
                     "/currsine1h":"get sine value of 1-hour full positive sine cycle",
                     "/currsine24h":"get sine value of 24-hour full positive sine cycle",
                  }
                }

@app.route('/')
def index():
    return supported_calls

def deserializer(item,observations):
    print('in deserializer '+str(item))
    item=item.replace("[","")
    item=item.replace("]","")
    print('item='+str(item))
    for num in item.split(" "):
       if (num):
         print('num='+str(num))
         observations.append(float(num))

def get_last_obs_arr(table):
    print('in get_last_obs_arr {}'.format(table))
    observations=[]
    try:
      response = table.get_item(
      Key={
        'key': 'observation'
      }
      )
      x=json.dumps(response['Item'])
      item = response['Item']['value']
      deserializer(item,observations)
      print('observations='+str(observations))
      return observations
    except Exception as e:
      print('Error {}'.format(e))

def put_latest_gs_inference(value,table):
    print ('in put_latest_gs_inference='+str(value))
    observations=get_last_obs_arr(table)
    print('observations before {}'.format(observations))
    try:
      observations=observations[1:]
      observations=np.append(observations,float(value))
      observations=str(observations)
      print ('observations after appending {}'.format(observations))
    except Exception as e:
      print('Error {}'.format(e))
    table.put_item(
       Item={
          'key': 'observation',
          'value': observations
       }
    )
    return observations

def populate_cloudwatch_metric(namespace,metric_value,metric_name):
    print("in populate_cloudwatch_metric metric_value="+str(metric_value)+" metric_name="+metric_name)
    response = cloudwatch_cli.put_metric_data(
        Namespace=namespace,
        MetricData=[
           {
              'MetricName': metric_name,
              'Unit': 'None',
              'Value': metric_value,
           },
        ]
        )
    print('response from cloud watch'+str(response))

@app.route('/predict/{region}')
def get_prediction(region):
    print ('in predict for region {}'.format(region))
    dynamodb = boto3.resource('dynamodb',region_name=region)
    latest_observations_table = dynamodb.Table('latest_observations')
    action=0

    print('get the last observations')
    payload = get_last_obs_arr(latest_observations_table)
    last_observations = json.dumps(payload)
    print('the last observations are {}'.format(last_observations))
    print('get the current demand - the number active game servers that holds active sessions')
    try:
      #we get curr_demand from external endpoint denoted by gs_inventory_url. To simplfy things we make a local call to help function get_curr_sine1h() instead. In real life, uncomment the four lines below to populate authentic curr_demand
      gs_url=gs_inventory_url
      req=requests.get(url=gs_url)
      data=req.json()
      #data=get_curr_sine1h()
      curr_demand = float(data['Prediction']['num_of_gameservers'])
      print('the current demand is {}'.format(curr_demand))

    except requests.exceptions.RequestException as e:
      print(e)
      print('if matchmaking did not respond just randomized curr_demand between limit, reward will correct')

    print('get the prediction from the SageMaker hosted endpoint')
    response = sagemaker_client.invoke_endpoint(
      EndpointName=endpoint_name,
      ContentType=content_type,
      Accept=accept,
      Body=last_observations
    )
    result = json.loads(response['Body'].read().decode())
    print('prediction results from Sagemaker {}'.format(result))
    action=float(result['predictions'][0]['actions'][0])
    action*=100
    action=np.clip(action,min_servers,max_servers)
    print('normelized action={}'.format(action))

    print('calculating the current prediction')
    print('needed {}; predicted {}'.format(curr_demand,action))
    print('checking if false-positive')
    new_alloc_norm=0
    if (action<curr_demand):
       print('false-positive; needed {}; predicted {}'.format(action,curr_demand))
       is_false_positive=1
       populate_cloudwatch_metric(namespace,1.0,'false-positive')
       new_alloc=curr_demand
       new_alloc_norm=curr_demand
    else:
       print('true-positive; needed {}; predicted {}'.format(action,curr_demand))
       #new_alloc_norm=(action+new_alloc_estimate)/2
       new_alloc_norm=action
       new_alloc=action
       is_false_positive=0

    print('instrumenting in cloudwatch')
    populate_cloudwatch_metric(namespace,curr_demand,'curr_demand')
    populate_cloudwatch_metric(namespace,new_alloc,'curr_alloc')
    populate_cloudwatch_metric(namespace,new_alloc_norm,'curr_alloc_norm')

    print('store the last predictions')
    #last_observations=put_latest_gs_inference(curr_demand,last_observations,latest_observations_table)
    last_observations=put_latest_gs_inference(curr_demand,latest_observations_table)

    print('sending back current prediction')
    return {"Prediction":{
              "num_of_gameservers": new_alloc,
              "observations":str(last_observations),
              "raw_results":result,
              "curr_demand":str(curr_demand),
              "is_false_positive":is_false_positive,
              "action":str(action),
              "new_alloc":str(new_alloc)
           }}

@app.route('/currtime')
def get_curr_time():
    current_time=strftime("%H:%M", gmtime())
    return {"Current time in the cloud":str(current_time)}

@app.route('/currsine24h')
def get_curr_sine24h():
    #length of full cycle in sec
    #10 hours * 60 min * 60 sec
    num_of_points_in_cycle=36000
    begin_point=0.2
    end_point=3.1
    factor=99

    current_hour=int(strftime("%H", gmtime()))
    cycle_arr=np.linspace(begin_point,end_point,num_of_points_in_cycle)
    current_point_in_cycle=int(time.time())%num_of_points_in_cycle

    print('current_hour='+str(current_hour))
    rand_hour=np.random.randint(10,15)
    print('rand_hour='+str(rand_hour))
    #inducing chaos every time the current hour is a randomnumber between 10 to 15
    if (current_hour==rand_hour):
        n=np.random.randint(25,100)
    else:
      current_point=cycle_arr[int(current_point_in_cycle)]
      print('current_point='+str(current_point))
      sine=factor*np.sin(current_point)
      print('sine='+str(sine))
      n=sine

    print('n='+str(n))
    return {"Prediction":{"num_of_gameservers": n}}

@app.route('/currsine1h')
def get_curr_sine1h():
    cycle_arr=sinearr=np.linspace(0.2,3.1,61)
    current_min=strftime("%M", gmtime())
    print("current_min="+str(current_min))

    current_point=cycle_arr[int(current_min)]
    print("current_point="+str(current_point))
    sine=99*np.sin(current_point)
    print("sine="+str(sine))
    return {"Prediction":{"num_of_gameservers": sine}}

