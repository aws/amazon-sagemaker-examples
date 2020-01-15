import time
import boto3
import requests
import gym
import numpy as np
from time import gmtime,strftime
from gym.spaces import Discrete, Box

cloudwatch_cli = boto3.client('cloudwatch',region_name='us-west-2')
 
class GameServerEnv(gym.Env):

    def __init__(self, env_config={}):
        print ("in __init__")
        print ("env_config {}".format(env_config))
        self.namespace = env_config['cloudwatch_namespace']
        self.gs_inventory_url = env_config['gs_inventory_url']
        self.learning_freq = env_config['learning_freq']
        self.min_servers = int(env_config['min_servers'])
        self.max_servers = int(env_config['max_servers'])
        self.action_factor = int(env_config['action_factor'])
        self.over_prov_factor = int(env_config['over_prov_factor'])
        self.num_steps = 0
        self.max_num_steps = 301
        self.history_len = 5
        self.total_num_of_obs = 1
        # we have two observation array, allocation and demand. allocation is alloc_observation, demand is observation hence *2
        self.observation_space = Box(low=np.array([self.min_servers]*self.history_len*2),
                                           high=np.array([self.max_servers]*self.history_len*2),
                                           dtype=np.uint32)
        
        # How many servers should the agent spin up at each time step 
        self.action_space = Box(low=np.array([0]),
                                     high=np.array([1]),
                                     dtype=np.float32)

    def reset(self):
        print ("in reset")
        #self.populate_cloudwatch_metric(self.namespace,1,'reset')
        self.num_steps = 0
        self.current_min = 0
        self.demand_observation = np.array([self.min_servers]*self.history_len)
        self.alloc_observation = np.array([self.min_servers]*self.history_len)
        
        print ('self.demand_observation '+str(self.demand_observation))
        print ('self.alloc_observation '+str(self.alloc_observation))
        return np.concatenate((self.demand_observation, self.alloc_observation))

   

    def step(self, action):
        print ('in step - action recieved from model'+str(action))
        self.num_steps+=1
        self.total_num_of_obs+=1
        print('total_num_of_obs={}'.format(self.total_num_of_obs))

        raw_action=float(action)
        self.curr_action = raw_action*self.action_factor
        self.curr_action = np.clip(self.curr_action, self.min_servers, self.max_servers)
        print('self.curr_action={}'.format(self.curr_action))
        
               
        if (self.gs_inventory_url!='local'):
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
        if (self.gs_inventory_url=='local'):
          print('local matchmaking service for current demand, curr_demand')
          data=self.get_curr_sine1h()
          self.curr_demand = float(data['Prediction']['num_of_gameservers'])       
        # clip the demand to the allowed range
        self.curr_demand = np.clip(self.curr_demand, self.min_servers, self.max_servers)
        print('self.curr_demand={}'.format(self.curr_demand)) 

        #time-horizon - use the oldest observation for current allocation
        self.curr_alloc = self.alloc_observation[0]
        print('self.curr_alloc={}'.format(self.curr_alloc)) 
            
        # Assumes it takes history_len time steps to create or delete 
        # the game server from allocation
        # self.action_observation = self.action_observation[1:]
        # self.action_observation = np.append(self.action_observation, self.curr_action)
        # print('self.action_observation={}'.format(self.action_observation))
        
        # store the current demand in the history array demand_observation
        self.demand_observation = self.demand_observation[1:] # shift the observation by one to remove one history point
        self.demand_observation=np.append(self.demand_observation,self.curr_demand)
        print('self.demand_observation={}'.format(self.demand_observation))
        
        # store the current allocation in the history array alloc_observation
        self.alloc_observation = self.alloc_observation[1:] 
        self.alloc_observation=np.append(self.alloc_observation,self.curr_action)
        print('self.alloc_observation={}'.format(self.alloc_observation))
 
        
        #reward calculation - in case of over provision just 1-ratio. under provision is more severe so 500% more negative reward
        print('calculate the reward, calculate the ratio between allocation and demand, we use the first allocation in the series of history of five, first_alloc/curr_demand')
        print('history of previous predictions made by the model ={}'.format(self.alloc_observation))
        
        ratio=self.curr_alloc/self.curr_demand
        print('ratio={}'.format(ratio))
        if (ratio>1):
           #reward=1-ratio
           reward = -1 * (self.curr_alloc - self.curr_demand)
           print('reward over provision - ratio>1 - {}'.format(reward))
        if (ratio<1):
           #reward=-50*ratio
           reward = -5 * (self.curr_demand - self.curr_alloc) 
           print('reward under provision - ratio<1 - {}'.format(reward))
        if (ratio==1):
           reward=1
           print('ratio=1')
        reward -= (self.curr_demand - self.curr_alloc)*self.over_prov_factor
        print('ratio={}'.format(ratio))
        print('reward={}'.format(reward))
                
         
        #Instrumnet the supply and demand in cloudwatch
        print('populating cloudwatch - self.curr_demand={}'.format(self.curr_demand))
        self.populate_cloudwatch_metric(self.namespace,self.curr_demand,'curr_demand')
        print('populating cloudwatch - self.curr_alloc={}'.format(self.curr_action))
        self.populate_cloudwatch_metric(self.namespace,self.curr_action,'curr_alloc')
        print('populating cloudwatch - reward={}'.format(reward))
        self.populate_cloudwatch_metric(self.namespace,reward,'reward')
        
        if (self.num_steps >= self.max_num_steps):
          done = True
          print ("self.num_steps "+str(self.num_steps))
          print ("self.max_num_steps "+str(self.max_num_steps))
        else:
          done = False
        
        print ('time.sleep() for {} before next iteration'.format(self.learning_freq))
        time.sleep(int(self.learning_freq)) 
        
        extra_info = {}
        #the next state includes the demand and allocation history. 
        next_state=np.concatenate((self.demand_observation,self.alloc_observation))
        print ('next_state={}'.format(next_state))
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
        
    def get_curr_sine1h(self):
        max_servers=self.max_servers*0.9
        print ('in get_curr_sine1h')
        cycle_arr=np.linspace(0.2,3.1,61)
        self.current_min = (self.current_min + 1) % 60
        current_min = self.current_min
        print('current_min={}'.format(current_min))
        current_point=cycle_arr[int(current_min)]
        sine=max_servers*np.sin(current_point)
        print('sine({})={}'.format(current_point,sine))
        return {"Prediction":{"num_of_gameservers": sine}}


