import os
import socket

import gym
import numpy as np
from eplus.envs import pyEp
from eplus.envs.socket_builder import socket_builder
from gym import error, spaces, utils
from gym.utils import seeding


class LargeOfficeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):

        # timestep=12, days=1, eplus_path=None,
        #         weather_file = 'weather/SPtMasterTable_587017_2012_amy.epw'):

        cur_dir = os.path.dirname(__file__)
        # print("File directory: ", cur_dir)

        # buildings/LargeOffice/LargeOfficeFUN.idf is the EnergyPlus file
        # used for this environment. The LargeOffice folder also contains
        # variables.cfg which configures the external input and output
        # variables
        self.idf_file = cur_dir + "/buildings/LargeOffice/LargeOfficeFUN.idf"

        # EnergyPlus weather file
        if "weather_file" in config:
            self.weather_file = cur_dir + "/" + config["weather_file"]
        else:
            self.weather_file = cur_dir + "/weather/SPtMasterTable_587017_2012_amy.epw"
        # self.weather_file = cur_dir + '/weather/SPtMasterTable_587017_2012_amy.epw'

        if "eplus_path" in config:
            self.eplus_path = config["eplus_path"]
        else:
            # Using EnergyPlus version 8.80, path to the executable
            # Assuming Mac
            self.eplus_path = "/Applications/EnergyPlus-8-8-0/"

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
        self.deltaT = (60 / self.epTimeStep) * 60

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
        # self.observation_space = spaces.Tuple((
        #                              spaces.Box(low=0, high=10000000, shape=(1,),dtype=np.float32), #0: facility total electric power (W)
        #                              spaces.Box(low=0, high=1000000, shape=(1,),dtype=np.float32), #1: current time of day
        #                              spaces.Box(low=0, high=7, shape=(1,),dtype=np.uint8), #2: current day of week
        #                              spaces.Box(low=0, high=30, shape=(1,),dtype=np.float32), #3: chiller 1 outlet temperature
        #                              spaces.Box(low=0, high=30, shape=(1,),dtype=np.float32), #4: chiller 2 outlet temperature
        #                              spaces.Box(low=30, high=150, shape=(1,),dtype=np.float32), #5: boiler outlet temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #6: basement temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #7: core bottom temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #8: core mid temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #9: core top temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #10: ground floor plenum temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #11: mid-floor plenum temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #12: zone 1 bottom temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #13: zone 2 bottom temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #14: zone 3 bottom temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #15: zone 4 bottom temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #16: zone 1 mid temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #17: zone 2 mid temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #18: zone 3 mid temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #19: zone 4 mid temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #20: zone 1 top temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #21: zone 2 top temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #22: zone 3 top temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #23: zone 4 top temperature
        #                              spaces.Box(low=0, high=60, shape=(1,),dtype=np.float32), #24: top floor plenum zone temperature
        #                              spaces.Box(low=-50, high=70, shape=(1,),dtype=np.float32), #25: outdoor drybulb temperature
        #                              spaces.Box(low=-50, high=70, shape=(1,),dtype=np.float32), #26: outdoor wetbulb temperature (C)
        #                              spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32), #27: wind speed (m/s)
        #                              spaces.Box(low=0, high=360, shape=(1,),dtype=np.float32), #28: wind direction (deg)
        #                              spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32), #29: outdoor relative humidity (%)
        #                              spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32), #30: heating setpoint (C)
        #                              spaces.Box(low=0, high=100, shape=(1,),dtype=np.float32), #31: cooling setpoint (C)
        #                              spaces.Box(low=0, high=300, shape=(1,),dtype=np.float32)  #32: hot water loop setpoint (C)
        #                                         ))
        self.observation_space = spaces.Box(
            np.array(
                [0, 0, 0, -50]
            ),  # chiller outlet temp, boiler outlet, zone 2 mid, outdoor temp
            np.array([30, 150, 60, 70]),
            dtype=np.float32,
        )

        # actions are all the control inputs
        # self.action_space = spaces.Tuple(( #spaces.Box(low=22, high=27, shape=(1,),dtype=np.float32), #cooling setpoint
        #                                  spaces.Box(low=6, high=7, shape=(1,),dtype=np.float32), #chiller setpoint
        #                                  spaces.Box(low=0, high=1, shape=(1,),dtype=np.float32)  #lighting setpoint
        #                                    ))
        self.chlr_min = 6  # chiller setpoint min in celcius
        self.chlr_max = 7  # chiller setpoint max in celcius
        self.light_min = 0  # normalized lighting level min
        self.light_max = 1  # normalized lighting level max
        self.action_space = spaces.Box(
            np.array([self.chlr_min, self.light_min]),
            np.array([self.chlr_max, self.light_max]),
            dtype=np.float32,
        )

    def step(self, action):

        # while(self.kStep < self.MAXSTEPS):
        # current time from start of simulation
        time = self.kStep * self.deltaT

        # current time from start of day
        dayTime = time % 86400

        if dayTime == 0:
            print("Day: ", int(self.kStep / self.DAYSTEPS) + 1)

        # inputs should be same as actions
        # force action to be within limits
        chiller_setpoint = np.clip(action, self.chlr_min, self.chlr_max)[0]
        lighting_level = np.clip(action, self.light_min, self.light_max)[1]
        self.inputs = [chiller_setpoint, lighting_level]
        input_packet = self.ep.encode_packet_simple(self.inputs, time)
        self.ep.write(input_packet)

        # after EnergyPlus runs the simulation step, it returns the outputs
        output_packet = self.ep.read()
        self.outputs = self.ep.decode_packet_simple(output_packet)

        # reward needs to be a combination of energy and comfort requirement
        # just using negative of total energy as the reward for the time being
        energy_coeff = -0.00001
        heating_coeff = -100
        cooling_coeff = -100
        energy = self.outputs[0]
        zone_temperature = self.outputs[17]  # taking mid-zone 2 as an example
        heating_setpoint = self.outputs[30]
        cooling_setpoint = self.outputs[31]
        heating_penalty = max(heating_setpoint - zone_temperature, 0)
        cooling_penalty = max(zone_temperature - cooling_setpoint, 0)
        reward = (
            energy_coeff * energy
            + heating_coeff * heating_penalty
            + cooling_coeff * cooling_penalty
        )

        # state can be all the inputs required to make a control decision
        next_state = np.array(
            [self.outputs[3], self.outputs[5], self.outputs[17], self.outputs[25]]
        )
        # fake state space
        # next_state = np.array([3, 2, 1, 0])

        # print("Energy: ", energy, "Reward: ", reward, "Action: ", self.inputs)

        # increment simulation step count
        self.kStep += 1

        # done when number of steps of simulation reaches its maximum (e.g. 1 day)
        done = False
        if self.kStep >= (self.MAXSTEPS):
            # requires one more step to close the simulation
            input_packet = self.ep.encode_packet_simple(self.inputs, time)
            self.ep.write(input_packet)
            # output is empty in the final step
            # but it is required to read this output for termination
            output_packet = self.ep.read()
            last_output = self.ep.decode_packet_simple(output_packet)
            print("Finished simulation")
            done = True
            self.ep.close()
            self.ep = None

        # extra information we want to pass
        info = {}

        return next_state, reward, done, info

    def reset(self):
        # stop existing energyplus simulation
        if self.ep:
            print("Closing the old simulation and socket.")
            self.ep.close()  # needs testing: check if it stops the simulation
            self.ep = None

        # start new simulation
        print("Starting a new simulation for %d day(s)" % (self.simDays))
        self.kStep = 0
        idf_dir = os.path.dirname(self.idf_file)
        builder = socket_builder(idf_dir)
        configs = builder.build()
        self.ep = pyEp.ep_process(
            "localhost", configs[0], self.idf_file, self.weather_file, self.eplus_path
        )

        # read the initial outputs from EnergyPlus
        # these outputs are from warmup phase, so this does not count as a simulation step
        self.outputs = self.ep.decode_packet_simple(self.ep.read())

        return np.array([self.outputs[3], self.outputs[5], self.outputs[17], self.outputs[25]])
        # return np.array([3,2,1,0])

    def render(self, mode="human", close=False):
        pass
