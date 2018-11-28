import gym
from gym import error, spaces, utils
from gym.utils import seeding
from eplus.envs import pyEp
import socket
from eplus.envs.socket_builder import socket_builder
import numpy as np
import os

class DataCenterEnv(gym.Env):

    def __init__(self, config):
    
        #timestep=12, days=1, eplus_path=None,
        #         weather_file = 'weather/SPtMasterTable_587017_2012_amy.epw'):
        
        cur_dir = os.path.dirname(__file__)
        #print("File directory: ", cur_dir)
             
        # buildings/1ZoneDataCenter/1ZoneDataCenter.idf is the EnergyPlus file
        # used for this environment. The 1ZoneDataCenter folder also contains
        # variables.cfg which configures the external input and output 
        # variables
        self.idf_file = cur_dir + '/buildings/1ZoneDataCenter/1ZoneDataCenter.idf'

        # EnergyPlus weather file
        if "weather_file" in config:
            self.weather_file = cur_dir + '/' + config["weather_file"]
        else:
            self.weather_file = cur_dir + '/weather/SPtMasterTable_587017_2012_amy.epw'
        #self.weather_file = cur_dir + '/weather/SPtMasterTable_587017_2012_amy.epw'

        if "eplus_path" in config:
            self.eplus_path = config["eplus_path"]
        else:
            # Using EnergyPlus version 8.80, path to the executable
            # Assuming Mac
            self.eplus_path = '/Applications/EnergyPlus-8-8-0/'

        # EnergyPlus number of timesteps in an hour
        if "timestep" in config:
            self.epTimeStep = config["timestep"]
        else:
            self.epTimeStep = 12

        # EnergyPlus number of simulation days
        if "days" in config:
            self.simDays = config["days"]
        else:
            self.simDays = 1

       # Number of steps per day
        self.DAYSTEPS = int(24 * self.epTimeStep)

       # Total number of steps
        self.MAXSTEPS = int(self.simDays * self.DAYSTEPS)

        # Time difference between each step in seconds
        self.deltaT = (60/self.epTimeStep)*60

        # Outputs given by EnergyPlus, defined in variables.cfg
        self.outputs = []

        # Inputs expected by EnergyPlus, defined in variables.cfg
        self.inputs = []

        # Current step of the simulation
        self.kStep = 0

        # Instance of EnergyPlus simulation
        self.ep = None

        # state can be all the inputs required to make a control decision
        # getting all the outputs coming from EnergyPlus for the time being 
        self.observation_space = spaces.Box(np.array([0, -50, 0]), #zone temp, outdoor drybulb temp, relative humidity
                                            np.array([60, 70, 100]), dtype=np.float32)

        # actions are all the control inputs
        #self.action_space = spaces.Tuple(( #spaces.Box(low=22, high=27, shape=(1,),dtype=np.float32), #cooling setpoint
        #                                  spaces.Box(low=6, high=7, shape=(1,),dtype=np.float32), #chiller setpoint
        #                                  spaces.Box(low=0, high=1, shape=(1,),dtype=np.float32)  #lighting setpoint
        #                                    ))
        self.clg_min = 20 #cooling setpoint min in celcius
        self.clg_max = 35 #cooling setpoint max in celcius
        self.htg_min = 5  #heating setpoint min in celcius
        self.htg_max = 20 #heating setpoint max in celcius
        #self.action_space = spaces.Box(np.array([self.clg_min,self.htg_min]), 
        #                          np.array([self.clg_max, self.htg_max]), dtype=np.float32)
        # Normalized action space
        self.action_space = spaces.Box(np.array([0,0]), 
                                  np.array([1,1]), dtype=np.float32)


    def step(self, action):

        # while(self.kStep < self.MAXSTEPS):
        # current time from start of simulation
        time = self.kStep * self.deltaT
        
        # current time from start of day
        dayTime = time % 86400

        if dayTime == 0:
            print("Day: ", int(self.kStep/self.DAYSTEPS)+1)

        #inputs should be same as actions
        #bring the actions in the correct range
        #For Ray: assuming mean 0 and std dev 1 by ray
        #action[0] = action[0]*(self.clg_max - self.clg_min)+(self.clg_min+self.clg_max)/2.0
        #action[1] = action[1]*(self.htg_max - self.htg_min)+(self.htg_min+self.htg_max)/2.0

        #For Coach: input is 0 to 1 range
        action[0] = action[0]*(self.clg_max - self.clg_min)+(self.clg_min)
        action[1] = action[1]*(self.htg_max - self.htg_min)+(self.htg_min)


        #force action to be within limits
        cooling_setpoint = np.clip(action, self.clg_min, self.clg_max)[0]
        heating_setpoint = np.clip(action, self.htg_min, self.htg_max)[1]
        self.inputs = [cooling_setpoint, heating_setpoint]
        input_packet = self.ep.encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet) 

        #after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read() 
        self.outputs = self.ep.decode_packet_simple(output_packet)
        #print("Outputs:", self.outputs)
        if not self.outputs:
            print("Outputs:", self.outputs)
            print("Actions:", action)
            next_state = self.reset()
            return next_state, 0, False, {}

        # reward needs to be a combination of energy and comfort requirement
        energy_coeff = -0.00001
        heating_coeff = -100
        cooling_coeff = -100
        energy = self.outputs[0]
        zone_temperature = self.outputs[1] #taking mid-zone 2 as an example
        heating_setpoint = 15 #fixed lower limit in celcius 
        cooling_setpoint = 30 #fixed upper limit in celcius
        heating_penalty = max(heating_setpoint - zone_temperature, 0)
        cooling_penalty = max(zone_temperature - cooling_setpoint, 0)

        # punish if action is out of limits
        action_penalty_coeff = -100
        max_penalty = max(self.clg_min - action[0], 0)
        min_penalty = max(action[0] - self.clg_max, 0)
        action_penalty = action_penalty_coeff * (max_penalty + min_penalty)
        max_penalty = max(self.htg_min - action[1], 0)
        min_penalty = max(action[1] - self.htg_max, 0)
        action_penalty += action_penalty_coeff * (max_penalty + min_penalty)

        # final reward
        reward = energy_coeff * energy \
                    + heating_coeff * heating_penalty \
                    + cooling_coeff * cooling_penalty \
                    + action_penalty

        # state can be all the inputs required to make a control decision
        # zone temp, outside drybulb temp, outside wetbulb temp, relative humidity
        next_state = np.array([self.outputs[1], self.outputs[2], self.outputs[4]])
        # fake state space
        #next_state = np.array([3, 2, 1, 0])

        #print("energy: %.2f, reward: %.2f, action: %.2f, %.2f" \
        #                     % (energy, reward, action[0], action[1]))
        #print("zone temp: %.2f, drybulb: %.2f, humidity: %.2f"\
        #                        %tuple(next_state))

        # increment simulation step count
        self.kStep += 1

        # done when number of steps of simulation reaches its maximum (e.g. 1 day)
        done = False
        if self.kStep >= (self.MAXSTEPS):
            #requires one more step to close the simulation
            input_packet = self.ep.encode_packet_simple(self.inputs, time)
            self.ep.write(input_packet)
            #output is empty in the final step
            #but it is required to read this output for termination
            output_packet = self.ep.read() 
            last_output = self.ep.decode_packet_simple(output_packet)
            print("Finished simulation")
            print("Last action: ", action)
            print("Last reward: ", reward)
            done = True
            self.ep.close()
            self.ep = None

        # extra information we want to pass 
        info = {}
        # print("State:", next_state, "Reward:", reward)
        return next_state, reward, done, info

    def reset(self):
        # stop existing energyplus simulation
        if self.ep:
            print("Closing the old simulation and socket.")
            self.ep.close() #needs testing: check if it stops the simulation
            self.ep = None

        # start new simulation
        print("Starting a new simulation..")
        self.kStep = 0
        idf_dir = os.path.dirname(self.idf_file)
        builder = socket_builder(idf_dir)
        configs = builder.build()  
        self.ep = pyEp.ep_process('localhost', configs[0], self.idf_file, self.weather_file, self.eplus_path)

        # read the initial outputs from EnergyPlus
        # these outputs are from warmup phase, so this does not count as a simulation step
        self.outputs = self.ep.decode_packet_simple(self.ep.read())
        return np.array([self.outputs[1], self.outputs[2], self.outputs[4]])
        #return np.array([3,2,1,0])
        
    def render(self, mode='human', close=False):
        pass

