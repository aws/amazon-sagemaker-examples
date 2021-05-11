import itertools
import os
import queue
import time
import uuid
from queue import Queue
from threading import Event, Thread

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from pyenergyplus.api import EnergyPlusAPI


class MediumOfficeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config):
        cur_dir = os.path.dirname(__file__)
        self.idf_file = cur_dir + "/buildings/MediumOffice/RefBldgMediumOfficeNew2004_Chicago.idf"

        # EnergyPlus weather file
        if "weather_file" in config:
            self.weather_file = cur_dir + "/" + config["weather_file"]
        else:
            self.weather_file = cur_dir + "/weather/SPtMasterTable_587017_2012_amy.epw"

        self.eplus_path = config["eplus_path"]

        # EnergyPlus number of timesteps in an hour
        if "timestep" in config:
            self.epTimeStep = config["timestep"]
        else:
            self.epTimeStep = 1

        # EnergyPlus number of simulation days
        if "days" in config:
            self.simDays = config["days"]
        else:
            self.simDays = 1

        self.energy_temp_penalty_ratio = config["energy_temp_penalty_ratio"]
        self.multi_zone_control = config["multi_zone_control"]

        # Number of steps per day
        self.DAYSTEPS = int(24 * self.epTimeStep)

        # Total number of steps
        self.MAXSTEPS = int(self.simDays * self.DAYSTEPS)

        # Time difference between each step in seconds
        self.deltaT = (60 / self.epTimeStep) * 60

        # Current step of the simulation
        self.kStep = 0

        # actions are all the control inputs
        self.min_heat_setpoint = 0.15  # min HTGSETP_SCH
        self.max_heat_setpoint = 0.22  # max HTGSETP_SCH
        self.min_cool_setpoint = 0.22  # min CLGSETP_SCH
        self.max_cool_setpoint = 0.30  # max CLGSETP_SCH

        self.api = EnergyPlusAPI()

        self.zones = [
            "Core_bottom",
            "Core_mid",
            "Core_top",
            "Perimeter_bot_ZN_1",
            "Perimeter_bot_ZN_2",
            "Perimeter_bot_ZN_3",
            "Perimeter_bot_ZN_4",
            "Perimeter_mid_ZN_1",
            "Perimeter_mid_ZN_2",
            "Perimeter_mid_ZN_3",
            "Perimeter_mid_ZN_4",
            "Perimeter_top_ZN_1",
            "Perimeter_top_ZN_2",
            "Perimeter_top_ZN_3",
            "Perimeter_top_ZN_4",
        ]
        self.eplus_output_vars = (
            [("Air System Total Heating Energy", f"VAV_{i}") for i in range(1, 4)]
            + [
                (f"Site Outdoor Air {cond}bulb Temperature", "Environment")
                for cond in ["Dry", "Wet"]
            ]
            + [("Zone Mean Air Temperature", zone) for zone in self.zones]
        )

        self.eplus_actuator_vars = [
            ("Schedule:Constant", "Schedule Value", f"{level}_SCH_{zone}")
            for zone in self.zones
            for level in ["HTGSETP", "CLGSETP"]
        ]

        min_setpoints = [
            (self.min_heat_setpoint, self.min_cool_setpoint) for _ in self.eplus_actuator_vars
        ]
        max_setpoints = [
            (self.max_heat_setpoint, self.max_cool_setpoint) for _ in self.eplus_actuator_vars
        ]

        if self.multi_zone_control:
            self.action_space = spaces.Box(
                np.array(list(itertools.chain(*min_setpoints))),
                np.array(list(itertools.chain(*max_setpoints))),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                np.array([self.min_heat_setpoint, self.min_cool_setpoint]),
                np.array([self.max_heat_setpoint, self.max_cool_setpoint]),
                dtype=np.float32,
            )
        self.min_zone_temp = 0
        self.max_zone_temp = 0.40
        self.min_total_power = 0
        self.max_total_power = 1e8
        self.desired_zone_temp = 0.22

        self.observation_space = spaces.Box(
            np.array(
                [0, 0, 0, -0.5, -0.5] + [self.min_zone_temp for _ in self.eplus_output_vars[5:]]
            ),
            np.array(
                [1, 1, 1, 0.5, 0.5] + [self.max_zone_temp for _ in self.eplus_output_vars[5:]]
            ),
            dtype=np.float32,
        )

        self.prev_output = tuple(
            [0.5, 0.5, 0.5, 0.15, 0.15]
            + [
                np.mean([self.min_zone_temp, self.max_zone_temp])
                for _ in self.eplus_output_vars[5:]
            ]
        )
        self.prev_control = tuple([self.desired_zone_temp for _ in self.zones])

        self.eplus_th = None

        self.control_queue = Queue()
        self.output_queue = Queue()

        self.simulation_complete = False
        self.is_data_ready = False
        self.is_warmup_done = False

    def flush_queues(self):
        for q in [self.control_queue, self.output_queue]:
            while not q.empty():
                q.get()

    def __setup_eplus_simulator(self):
        self.is_data_ready = False

        if self.eplus_th is not None:
            self.simulation_complete = True
            self.eplus_th.join()

        self.api.api.cClearAllStates()

        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.__read_energyplus_output
        )

        [self.api.exchange.request_variable(v[0], v[1]) for v in self.eplus_output_vars]

        self.flush_queues()

        self.eplus_th = Thread(
            target=self.api.runtime.run_energyplus, args=(["-w", self.weather_file, self.idf_file],)
        )
        self.eplus_th.start()

        self.simulation_complete = False
        self.first_time_in_progress_handler = False

    def __check_if_data_ready(self):
        if self.is_data_ready:
            return True
        elif self.api.exchange.api_data_fully_ready():
            self.output_handles = [
                self.api.exchange.get_variable_handle(v[0], v[1]) for v in self.eplus_output_vars
            ]
            self.actuator_handles = [
                self.api.exchange.get_actuator_handle(v[0], v[1], v[2])
                for v in self.eplus_actuator_vars
            ]

            if -1 in self.actuator_handles:
                print(f"Actuator handles, {self.actuator_handles}")
                raise SystemExit("Could not get actuator handles.")

            if -1 in self.output_handles:
                print(f"Output handles, {self.output_handles}")
                raise SystemExit("Could not get output handles.")

            self.is_data_ready = True

        return self.is_data_ready

    def __read_energyplus_output(self):
        if (
            self.simulation_complete
            or self.api.exchange.warmup_flag()
            or not self.__check_if_data_ready()
        ):
            return

        output = tuple(self.api.exchange.get_variable_value(h) for h in self.output_handles)
        normalized_output = [p / self.max_total_power for p in output[0:3]] + [
            t / 100 for t in output[3:]
        ]
        self.prev_output = normalized_output

        self.output_queue.put(normalized_output)

        try:
            control = self.control_queue.get(timeout=10)
            self.prev_control = control
        except queue.Empty:
            print("Control queue empty. Using default values.")
            control = self.prev_control

        for ind in range(len(self.actuator_handles)):
            self.api.exchange.set_actuator_value(self.actuator_handles[ind], control[ind] * 100)

    def step(self, action):
        # current time from start of simulation
        cur_time = self.kStep * self.deltaT

        # current time from start of day
        if cur_time % 86400 == 0:
            print(f"Day: {int(self.kStep/self.DAYSTEPS) + 1}")

        #         action_zones = [action] * len(self.zones)
        self.control_queue.put(action)

        try:
            output = self.output_queue.get(timeout=10)
            self.past_output = output
        except queue.Empty:
            print("Output queue empty in step. Using past values.")
            output = self.prev_output

        # reward needs to be a combination of energy and comfort requirement
        energy_coeff = -1
        temp_coeff = self.energy_temp_penalty_ratio * energy_coeff

        energy = sum(output[0:3]) / self.deltaT
        temp_penalty = np.mean([abs(self.desired_zone_temp - t) for t in output[5:]])

        reward = energy_coeff * energy + temp_coeff * temp_penalty

        # state can be all the inputs required to make a control decision
        next_state = np.array(output)

        #         print(f'Energy: {energy}, Reward: {reward}, Action: {action}, Next state: {next_state}')

        # increment simulation step count
        self.kStep += 1

        # done when number of steps of simulation reaches its maximum (e.g. 1 day)
        done = False
        if self.kStep >= (self.MAXSTEPS):
            print("Finished simulation.")
            done = True

        # extra information we want to pass
        info = {}

        return next_state, reward, done, info

    def reset(self):
        # stop existing energyplus simulation and start new simulation
        print(f"Starting a new simulation for {self.simDays} day(s)")
        self.kStep = 0

        self.__setup_eplus_simulator()

        try:
            output = self.output_queue.get()
        except queue.Empty:
            print("Output queue empty in reset. Using past values.")
            output = self.prev_output

        return np.array(output)

    def render(self, mode="human", close=False):
        pass
