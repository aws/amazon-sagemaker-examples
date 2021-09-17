import gym
import numpy as np
from gym.envs.registration import EnvSpec
from gym.spaces import Box, Discrete
from gymhelper import box_space_from_description


class MultiEma:
    """Utility class that handles multiple streams of exponential moving average (EMA)
    with multiple different alpha values.
    """

    def __init__(self, num_streams, alphas):
        self.alphas = np.asarray(alphas)
        self.num_alphas = self.alphas.shape[0]
        self.num_streams = num_streams
        self.values = np.zeros(shape=(self.num_streams, self.num_alphas))

    def update(self, current):
        """Takes a vector of current observations.  Updates all the EMA streams,
        and returns them in a (N,A) matrix for the N streams and A alpha values.
        """
        decayed = self.values * (1.0 - self.alphas)
        updates = np.outer(np.asarray(current), self.alphas)
        self.values = decayed + updates
        return self.values


class SimpleScalableWebserviceSim(gym.Env):
    """A simple simulator of a scalable web service, including the following features:
    - Variable simulated load from customers
    - Simple financial model for the reward
        - Costs for running machines
        - Value for completing successful transactions
        - Big penalty for insufficience capacity or "downtime"
    - Delay to turn on & warm up new machines
    """

    def __init__(self, **config):
        config_defaults = {
            "warmup_latency": 5,  # It takes 5 minutes for a new machine to warm up and become available.
            "tpm_per_machine": 300,  # Each machine can process 300 transactions per minute (tpm) on average
            "tpm_sigma": 30,  # Machine's TPM capacity is variable with +/- 30 standard deviation
            "machine_cost": 0.05,  # Machines cost $0.05/min
            "transaction_val": 0.90,  # Successful transactions are worth $0.90 per thousand (CPM)
            "downtime_cost": 200,  # Downtime is assumed to cost the business $200/min beyond incomplete transactions
            "downtime_percent": 99.5,  # Downtime is defined as availability dropping below 99.5%
            "initial_machines": 50,  # How many machines are initially turned on
            "max_time_steps": 10000,  # Maximum number of timesteps per episode
        }

        for key, val in config_defaults.items():
            val = config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[
                key
            ] = val  # Creates variables like self.tpm_per_machine, self.tpm_sigma, etc

        # Internal "system" limits
        self.max_tpm = 1e5
        self.max_machines = int(2.5 * self.max_tpm / self.tpm_per_machine)
        self.max_machine_delta = 10

        self.action_space = Box(low=0, high=self.max_machine_delta, shape=(2,), dtype=np.int8)
        """At each time step, the agent is allowed to add machines AND substract machines.  It can do both at the same time,
        which isn't necessarily a good idea, but probably actually makes it easier to learn a good policy, because
        then the default do-nothing state is to have 0 for both outputs, which is very easy to learn with a ReLU output on the
        policy network.  In contrast, asking for a single regression output which is often 0 but sometimes a large value is harder to learn
        because it's like balancing on a knife-edge for that final neuron.
        """

        alphas = [1.0, 0.1, 0.01, 0.001, 0.0001]
        # alphas = [1.0]
        num_alphas = len(alphas)

        descriptions = []
        ema_descriptions = [
            ("load", 0, self.max_tpm, "Current load (transactions this minute)"),
            ("fail", 0, self.max_tpm, "Number of failed transactions this minute"),
            ("down", 0, 1, "Are we in downtime?"),
            ("active", 0, self.max_machines, "Current number of active machines"),
        ]
        self._ema = MultiEma(len(ema_descriptions), alphas)
        descriptions += ema_descriptions * num_alphas
        descriptions += [
            (
                "warmup%d" % m,
                0,
                self.max_machine_delta,
                "Number of machines that will be available in %d minutes" % m,
            )
            for m in range(1, self.warmup_latency + 1)
        ]

        self.observation_space = box_space_from_description(descriptions)

        self.load_simulator = LoadSimulator(self.max_tpm)
        self._spec = EnvSpec("SimpleScalableWebserviceSim-v0")
        self.reset()

    def reset(self):
        self.load_simulator.reset()
        self.active_machines = self.initial_machines
        self.warmup_queue = [0] * self.warmup_latency
        self.current_load = 0
        self._react_to_load()
        self.t = 0
        return self._observation()

    def _observation(self):
        # Scale everything to fit in a 0-1 scale to keep the NN problem well conditioned.
        new_ema_obs = [
            self.current_load / self.max_tpm,
            self.failed / (self.current_load + 1e-5),
            self.is_down,
            self.active_machines / self.max_machines,
        ]
        observation = (self._ema.update(new_ema_obs).T).ravel().tolist()
        observation += [x / self.max_machine_delta for x in self.warmup_queue]
        return observation

    def _react_to_load(self):
        """Returns reward.  Also updates internal state for _observation()"""
        self.capacity = int(
            self.active_machines * np.random.normal(self.tpm_per_machine, self.tpm_sigma)
        )
        if self.current_load <= self.capacity:
            # All transactions succeed
            self.failed = 0
            succeeded = self.current_load
        else:
            # Some transactions failed
            self.failed = self.current_load - self.capacity
            succeeded = self.capacity
        reward = succeeded * self.transaction_val / 1000.0  # divide by thousand for CPM
        percent_success = 100.0 * succeeded / (self.current_load + 1e-20)
        if percent_success < self.downtime_percent:
            self.is_down = 1
            reward -= self.downtime_cost
        else:
            self.is_down = 0
        reward -= self.active_machines * self.machine_cost
        return reward

    def step(self, action):
        # First, react to the actions and adjust the fleet
        turn_on_machines = int(action[0])
        turn_off_machines = int(action[1])
        self.active_machines = max(0, self.active_machines - turn_off_machines)
        warmed_up_machines = self.warmup_queue[0]
        self.active_machines = min(self.active_machines + warmed_up_machines, self.max_machines)
        self.warmup_queue = self.warmup_queue[1:] + [turn_on_machines]

        # Now react to the current load and calculate reward
        self.current_load = self.load_simulator.time_step_load()
        reward = self._react_to_load()

        self.t += 1
        done = self.t > self.max_time_steps

        # print("Machines %d+%d-%d.  Load: %d/%d.  Reward=%f" % (self.active_machines, turn_on_machines, turn_off_machines, self.capacity, self.current_load, reward))
        # print("Queue: %s" % str(self.warmup_queue))
        return self._observation(), reward, done, {}


class LoadSimulator:
    """Having a good simulation of the load over time is critical to the usefulness of this simulator.
    This is a pretty simple toy load simulator.  It has two components to load: periodic load and spikes.
    The periodic load is a simple daily cycle of fixed mean & amplitude, with multiplicative gaussian noise.
    The spike load start instantly and decay linearly until gone, and have a variable random delay between them.
    """

    def __init__(self, max_tpm=1e5):
        self.max_tpm = max_tpm
        self.reset()

    def reset(self):
        self.minutes = 0
        self.cyclic_min = np.random.uniform(1000, 2000)
        self.cyclic_max = self.cyclic_min + np.random.uniform(5000, 7000)
        self.cyclic_noise = np.random.uniform(0.01, 0.05)
        self.cyclic_phase = np.random.uniform(-np.pi, np.pi)
        # Add a low-frequency (LF) signal so it's not a perfect sinusoid
        self.lf_amp = np.random.uniform(0.1, 0.2)
        self.lf_period = np.random.uniform(2, 5)  # days
        self.lf_phase = np.random.uniform(-np.pi, np.pi)
        self._reset_spike()

    def _reset_spike(self):
        self.how_long_until_spike = np.random.uniform(100, 1000)
        self.spike_width = 5 + np.random.lognormal(3, 2)
        self.spike_height = self.cyclic_max * np.random.uniform(0.1, 0.2)

    def time_step_load(self):
        """External method that sanitizes the output"""
        self.minutes += 1
        load = int(self._calculate_load())
        load = max(load, 0)
        load = min(load, self.max_tpm)
        return load

    def _calculate_load(self):
        """The real algorithm for calculating load"""
        # First calculate the base daily-cycle load
        avg = (self.cyclic_min + self.cyclic_max) / 2.0
        amplitude = (self.cyclic_max - self.cyclic_min) / 2.0
        phase = self.minutes / (60 * 12) * np.pi + self.cyclic_phase
        cyclic_load = avg + amplitude * np.sin(phase)
        # add low-frequency signal
        cyclic_load *= 1 + np.sin(phase / self.lf_period + self.lf_phase) * self.lf_amp
        # add some noise
        cyclic_load *= np.random.normal(1, self.cyclic_noise)

        # Add a spike
        self.how_long_until_spike -= 1
        t = self.how_long_until_spike
        if t > 0:
            spike_load = 0
        else:
            # Spike is on!
            w = self.spike_width
            # Spike decays linearly
            spike_load = self.spike_height * (1.0 + t / w)
            if spike_load < 0:
                # This spike is over.
                self._reset_spike()  # Plan a new spike

        load = cyclic_load + spike_load
        return load
