import time
import boto3
import requests
import gym
import numpy as np
from time import gmtime,strftime
from gym.spaces import Discrete, Box

cloudwatch_cli = boto3.client('cloudwatch',region_name='us-west-2')
dynamodb = boto3.resource('dynamodb',region_name='us-west-2')
table = dynamodb.Table('observations')
 
class GameServerEnv(gym.Env):

    def __init__(self, env_config={}):
        print ("in __init__")
        print ("env_config")
        print (str(env_config))
        self.namespace = env_config['cloudwatch_namespace']
        self.gs_inventory_url = env_config['gs_inventory_url']
        self.learning_freq = env_config['learning_freq']
        self.min_servers = int(env_config['min_servers'])
        self.max_servers = int(env_config['max_servers'])
        self.num_steps = 0
        self.max_num_steps = 301
        self.demand_history_len = 5
        self.total_num_of_obs = 1
        self.observation_space = Box(low=np.array([self.min_servers]*self.demand_history_len),
                                           high=np.array([self.max_servers]*self.demand_history_len),
                                           dtype=np.uint32)
        
        # How many servers should the agent spin up at each time step 
        self.action_space = Box(low=np.array([0]),
                                     high=np.array([1]),
                                     dtype=np.float32)
        self.alloc_observation = np.array([self.min_servers]*self.demand_history_len) 
        self.populate_cloudwatch_metric(self.namespace,1,'init')

    def reset(self):
        print ("in reset")
        self.populate_cloudwatch_metric(self.namespace,1,'reset')
        self.num_steps = 0
        self.current_min = 0
        self.observation = np.array([self.min_servers]*self.demand_history_len)
        self.alloc_observation = np.array([self.min_servers]*self.demand_history_len)
        print ('self.observation '+str(self.observation))
        return self.observation

   

    def step(self, action):
        print ('in step - action recieved from model'+str(action))
        self.num_steps+=1
        self.total_num_of_obs+=1
        print('total_num_of_obs='+str(self.total_num_of_obs))

        raw_action=float(action)
        self.curr_alloc = raw_action*100
        
        print('clipping curr_alloc {} between {} and {}'.format(self.curr_alloc,self.min_servers,self.max_servers))
        clipped_action = np.clip(self.curr_alloc, self.min_servers, self.max_servers) 
        print ('np.clip clipped_action={}'.format(clipped_action))
        
        
        #get the demand from the matchmaking service
        print('quering matchmaking service for current demand, curr_demand')
        try:
           gs_url=self.gs_inventory_url
           req=requests.get(url=gs_url)
           data=req.json()
           self.curr_demand = float(data['Prediction']['num_of_gameservers'])            
            
        except requests.exceptions.RequestException as e:
           print(e)
           print('if matchmaking did not respond just randomized curr_demand between limit, reward will correct')
           self.curr_demand = float(np.random.randint(self.min_servers,self.max_servers))
        '''
        print('local matchmaking service for current demand, curr_demand')
        data=self.get_curr_sine1h()
        self.curr_demand = float(data['Prediction']['num_of_gameservers'])
        '''
        print('self.curr_demand={}'.format(self.curr_demand))
        
        
        #reward calculation - in case of over provision just 1-ratio. under provision is more severe so 500% more negative reward
        print('calculate the reward, calculate the ratio between allocation and demand, curr_alloc/curr_demand')
        ratio=self.curr_alloc/self.curr_demand
        print('interm ratio='+str(ratio))
        if (ratio>1):
           #reward=1-ratio
           reward = -1 * (self.curr_alloc - self.curr_demand)
           print('over provision - ratio>1 - '+str(reward))
        if (ratio<1):
           #reward=-50*ratio
           reward = -5 * (self.curr_demand - self.curr_alloc) 
           print('under provision - ratio<1 - '+str(reward))
        if (ratio==1):
           reward=1
           print('ratio=1')
        print('ratio='+str(ratio))
                
        # clip the demand to the allowed range
        self.curr_demand = np.clip(self.curr_demand, self.min_servers, self.max_servers)
        self.observation = self.observation[1:] # shift the observation by one to remove one history point
        self.observation=np.append(self.observation,self.curr_demand)
        
        print('self.observation '+str(self.observation))
        #self.put_latest_gs_inference(self.observation)
        next_state = self.observation
        
        self.alloc_observation = self.alloc_observation[1:] 
        self.alloc_observation=np.append(self.alloc_observation,self.curr_alloc)
        #is_new_min indicates if the current cycle is in new point on the minute. It is to be used with using local sine wave where the step execution frequancy is bounded to the instance clock cycle rather then self.learning_freq
        is_new_min=int(strftime("%S", gmtime()))
        print('is_new_min={}'.format(is_new_min))
        is_new_min=0
        if (is_new_min==0):
          print('Instrumnet the supply and demand in cloudwatch')
          print('populating cloudwatch - self.curr_demand={}'.format(self.curr_demand))
          self.populate_cloudwatch_metric(self.namespace,self.curr_demand,'curr_demand')
          print('populating cloudwatch - self.curr_alloc={}'.format(self.curr_alloc))
          self.populate_cloudwatch_metric(self.namespace,self.curr_alloc,'curr_alloc')
          print('populating cloudwatch - reward={}'.format(reward))
          self.populate_cloudwatch_metric(self.namespace,reward,'reward')
          print('populating cloudwatch - raw action={}'.format(raw_action))
          self.populate_cloudwatch_metric(self.namespace,raw_action,'raw_action')
        
          print('self.alloc_observation {}'.format(self.alloc_observation))
          normalized_alloc=np.percentile(self.alloc_observation,90)
          print('populating cloudwatch - normalized normalized_action={}'.format(normalized_alloc))
          self.populate_cloudwatch_metric(self.namespace,normalized_alloc,'normalized_alloc')
        
        if (self.num_steps >= self.max_num_steps):
          done = True
          print ("self.num_steps "+str(self.num_steps))
          print ("self.max_num_steps "+str(self.max_num_steps))
        else:
          done = False
        
        print ('time.sleep() for {} before next iteration'.format(self.learning_freq))
        time.sleep(int(self.learning_freq)) 
        
        extra_info = {}
        print ("next_state "+str(next_state))
        return next_state, reward, done, extra_info

    def render(self, mode):
        print("in render")
        pass

    def populate_cloudwatch_metric(self,namespace,metric_value,metric_name):
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
        
    def put_latest_gs_inference(self,observation_arr):
        print ('in put_latest_gs_inference='+str(observation_arr))
        observations=str(observation_arr)
        print ('observations='+observations)
        table.put_item(
          Item={
            'key': 'observation',
            'value': observations
          }
        )
    def get_curr_sine1h(self):
        print ('in get_curr_sine1h')
        cycle_arr=np.linspace(0.2,3.1,61)
        self.current_min = (self.current_min + 1) % 60
        current_min = self.current_min
        print('current_min={}'.format(current_min))
        current_point=cycle_arr[int(current_min)]
        sine=self.max_servers*np.sin(current_point)
        print('sine({})={}'.format(current_point,sine))
        return {"Prediction":{"num_of_gameservers": sine}}

