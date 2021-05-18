from math import isclose

import gym
import numpy as np
from gym.spaces import Box, Dict, Discrete
from scipy.stats import truncnorm

from utils import vrp_action_go_from_a_to_b

"""
STATE:
Restaurant x
Restaurant y
Driver x
Driver y
Driver used capacity
Driver max capacity
Order x
Order y
Order status: inactive/delivered/not-created, open, accepted, picked up
Order to restaurant map
Time elapsed per order after 'open'
Reward per order

ACTION:
Wait (Do nothing)
Accept order i
Pickup order i (by moving one step towards the respective restaurant)
Deliver order i (by moving one step towards the respective delivery location)
Go to restaurant i
"""


class VRPGymEnvironment(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 8}

    def render(self, mode="human", close=False):
        from vrp_view_2D import VRPView2D

        if self.vrp_view is None:
            self.vrp_view = VRPView2D(
                n_restaurants=self.n_restaurants,
                n_orders=self.n_orders,
                map_quad=self.map_quad,
                grid_size=30,
            )
        return self.vrp_view.update(
            res_x=self.res_x,
            res_y=self.res_y,
            o_status=self.o_status,
            o_x=self.o_x,
            o_y=self.o_y,
            dr_x=self.dr_x,
            dr_y=self.dr_y,
            o_res_map=self.o_res_map,
            mode=mode,
        )

    def __init__(self, env_config={}):

        self.vrp_view = None
        config_defaults = {
            "n_restaurants": 2,
            "n_orders": 10,
            "order_prob": 0.5,
            "driver_capacity": 4,
            "map_quad": (5, 5),
            "order_promise": 60,
            "order_timeout_prob": 0.15,
            "episode_length": 1000,
            "num_zones": 4,
            "order_probs_per_zone": (0.1, 0.5, 0.3, 0.1),
            "order_reward_min": (8, 5, 2, 1),
            "order_reward_max": (12, 8, 5, 3),
            "half_norm_scale_reward_per_zone": (0.5, 0.5, 0.5, 0.5),
            "penalty_per_timestep": 0.1,
            "penalty_per_move": 0.1,
            "order_miss_penalty": 50,
        }

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val
            if key not in env_config:
                env_config[key] = val

        assert len(self.order_probs_per_zone) == self.num_zones
        assert isclose(sum(self.order_probs_per_zone), 1.0)

        self.csv_file = "/opt/ml/output/data/vrp_rewards.csv"
        self.dr_used_capacity = 0
        self.o_x = []
        self.o_y = []
        self.o_status = []
        self.o_res_map = []
        self.o_time = []
        self.reward_per_order = []

        self.dr_x = None
        self.dr_y = None

        self.game_over = False
        self.state = []
        self.reward = None

        self.clock = 0

        # map boundaries
        self.map_min_x = -self.map_quad[0]
        self.map_max_x = +self.map_quad[0]
        self.map_min_y = -self.map_quad[1]
        self.map_max_y = +self.map_quad[1]
        self.map_range_x = range(-self.map_max_x, self.map_max_x + 1)
        self.map_range_y = range(-self.map_max_y, self.map_max_y + 1)

        # zone boundaries
        self.zone_range_x = np.array_split(np.array(self.map_range_x), self.num_zones)

        # restaurant x position limits
        res_x_min = [self.map_min_x] * self.n_restaurants
        res_x_max = [self.map_max_x] * self.n_restaurants
        # restaurant y position limits
        res_y_min = [self.map_min_y] * self.n_restaurants
        res_y_max = [self.map_max_y] * self.n_restaurants

        # driver x position limits
        dr_x_min = [self.map_min_x]
        dr_x_max = [self.map_max_x]
        # driver y position limits
        dr_y_min = [self.map_min_y]
        dr_y_max = [self.map_max_y]

        dr_used_capacity_min = [0]
        dr_used_capacity_max = [self.driver_capacity]

        # n_orders for x position limits
        o_x_min = [self.map_min_x] * self.n_orders
        o_x_max = [self.map_max_x] * self.n_orders
        # n_orders for y position limits
        o_y_min = [self.map_min_y] * self.n_orders
        o_y_max = [self.map_max_y] * self.n_orders

        # order status: 0 - inactive(not created, cancelled, delivered), 1 - open, 2 - accepted, 3 - picked-up
        o_status_min = [0] * self.n_orders
        o_status_max = [3] * self.n_orders

        # Reward per order
        reward_per_order_min = [0] * self.n_orders
        reward_per_order_max = [max(self.order_reward_max)] * self.n_orders

        # order-restaurant mapping, i.e. which the order belongs to which restaurant
        o_res_map_min = [-1] * self.n_orders
        o_res_map_max = [self.n_restaurants - 1] * self.n_orders

        # time elapsed since the order has been placed
        o_time_min = [0] * self.n_orders
        o_time_max = [self.order_promise] * self.n_orders

        # Create the observation space
        orig_observation_space = Box(
            low=np.array(
                res_x_min
                + res_y_min
                + dr_x_min
                + dr_y_min
                + dr_used_capacity_min
                + [self.driver_capacity]
                + o_x_min
                + o_y_min
                + o_status_min
                + o_res_map_min
                + o_time_min
                + reward_per_order_min
            ),
            high=np.array(
                res_x_max
                + res_y_max
                + dr_x_max
                + dr_y_max
                + dr_used_capacity_max
                + [self.driver_capacity]
                + o_x_max
                + o_y_max
                + o_status_max
                + o_res_map_max
                + o_time_max
                + reward_per_order_max
            ),
            dtype=np.int16,
        )
        # number of possible actions
        # Wait, Accept Order i, pick up order i, deliver order i, return to restaurant j
        self.max_avail_actions = 1 + 3 * self.n_orders + self.n_restaurants
        self.observation_space = Dict(
            {
                # a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
                "action_mask": Box(0, 1, shape=(self.max_avail_actions,), dtype=np.float32),
                "real_obs": orig_observation_space,
            }
        )
        self.action_space = Discrete(self.max_avail_actions)

    def reset(self):
        self.clock = 0

        self.__place_restaurants()
        self.__place_driver()
        self.dr_used_capacity = 0
        self.o_x = [0] * self.n_orders
        self.o_y = [0] * self.n_orders
        self.o_status = [0] * self.n_orders
        self.o_res_map = [-1] * self.n_orders
        self.o_time = [0] * self.n_orders
        self.reward_per_order = [0] * self.n_orders

        self.vrp_view = None

        return self.__reset_state()

    def step(self, action):
        orig_obs, rew, done, info = self.__orig_step(action)
        self.__update_avail_actions()
        obs = {
            "action_mask": self.action_mask,
            "real_obs": orig_obs,
        }
        return obs, rew, done, info

    def __orig_step(self, action):

        done = False
        self.info = {}
        self.reward = -self.penalty_per_timestep
        self.late_penalty = 0

        a = [self.dr_x, self.dr_y]

        action_type = None
        translated_action = None
        relevant_order_index = None

        if action == 0:  # Wait
            action_type = "wait"
        elif action <= self.n_orders:  # Accept an order
            action_type = "accept"
            relevant_order_index = action - 1
        elif (
            action <= 2 * self.n_orders
        ):  # Pick up a specific order (and go to the corresponding restaurant for that)
            relevant_order_index = action - self.n_orders - 1
            action_type = "pickup"
            res_ordered_from = self.o_res_map[relevant_order_index]
            b = [self.res_x[res_ordered_from], self.res_y[res_ordered_from]]
            translated_action = vrp_action_go_from_a_to_b(a, b)
            self.reward -= self.penalty_per_move
        elif action <= 3 * self.n_orders:  # Deliver the order
            relevant_order_index = action - 2 * self.n_orders - 1
            action_type = "deliver"
            b = [self.o_x[relevant_order_index], self.o_y[relevant_order_index]]
            translated_action = vrp_action_go_from_a_to_b(a, b)
            self.reward -= self.penalty_per_move
        elif action <= 3 * self.n_orders + self.n_restaurants:  # Return to a restaurant
            action_type = "return"
            destination_res = action - 3 * self.n_orders - 1
            b = [self.res_x[destination_res], self.res_y[destination_res]]
            translated_action = vrp_action_go_from_a_to_b(a, b)
            self.reward -= self.penalty_per_move
        else:
            raise Exception(
                "Misaligned action space and step function for action {}".format(action)
            )

        self.__update_driver_parameters(action_type, translated_action, relevant_order_index)
        self.__update_environment_parameters()
        state = self.__create_state()

        # Update the clock
        self.clock += 1
        if self.clock >= self.episode_length:
            done = True

        self.info["no_late_penalty_reward"] = self.reward + self.late_penalty

        return state, self.reward, done, self.info

    def __update_avail_actions(self):
        self.action_mask = np.array([0.0] * self.action_space.n)
        assert len(self.action_mask) == self.max_avail_actions
        # define & update invalid actions
        # always allow "wait" and "return to restaurant"
        self.action_mask[0] = 1.0
        self.action_mask[
            (3 * self.n_orders + 1) : (3 * self.n_orders + self.n_restaurants + 1)
        ] = 1.0

        for order_status_index in range(len(self.o_status)):
            if self.o_status[order_status_index] == 1:  # open order
                self.action_mask[order_status_index + 1] = 1.0  # allow "accept"
            elif (
                self.o_status[order_status_index] == 2
                and self.dr_used_capacity < self.driver_capacity
            ):  # accepted order
                relevant_order_index = order_status_index + self.n_orders + 1
                self.action_mask[relevant_order_index] = 1.0  # allow "pickup"
            elif self.o_status[order_status_index] == 3:  # picked up order
                relevant_order_index = order_status_index + 2 * self.n_orders + 1
                self.action_mask[relevant_order_index] = 1.0  # allow "deliver"

    def __update_dr_xy(self, a):
        if a == 1:  # UP
            self.dr_y = min(self.map_max_y, self.dr_y + 1)
        elif a == 2:  # DOWN
            self.dr_y = max(self.map_min_y, self.dr_y - 1)
        elif a == 3:  # LEFT
            self.dr_x = max(self.map_min_x, self.dr_x - 1)
        elif a == 4:  # RIGHT
            self.dr_x = min(self.map_max_x, self.dr_x + 1)

    def __update_driver_parameters(self, action_type, translated_action, relevant_order_index):
        if action_type == "wait":
            pass  # no action

        elif action_type == "accept":
            # if order accept it
            if self.o_status[relevant_order_index] == 1:
                self.o_status[relevant_order_index] = 2
                self.reward += (
                    self.reward_per_order[relevant_order_index] / 3
                )  # Give some reward for accepting

        elif action_type == "pickup":
            self.__update_dr_xy(translated_action)
            rix = self.o_res_map[relevant_order_index]
            if [self.dr_x, self.dr_y] == [self.res_x[rix], self.res_y[rix]]:
                if self.o_status[relevant_order_index] == 2:
                    self.o_status[relevant_order_index] = 3
                    self.dr_used_capacity += 1
                    self.reward += (
                        self.reward_per_order[relevant_order_index] / 3
                    )  # Give some reward for pickup

        elif action_type == "deliver":
            self.__update_dr_xy(translated_action)
            # Check for deliveries
            for o in range(self.n_orders):
                # If order is picked up by driver and driver is at delivery location, deliver the order
                if self.o_status[o] == 3 and (
                    self.dr_x == self.o_x[o] and self.dr_y == self.o_y[o]
                ):
                    if self.o_time[o] <= self.order_promise:
                        self.reward += (
                            self.reward_per_order[o] / 3
                        )  # Rest of the reward was given in accept and pickup
                    self.dr_used_capacity -= 1
                    self.__reset_order(o)
        elif action_type == "return":
            self.__update_dr_xy(translated_action)

        else:
            raise Exception(
                "Misaligned action space and driver update function: {}, {}, {}".format(
                    action_type, translated_action, relevant_order_index
                )
            )

    def __update_environment_parameters(self):
        # Update the waiting times
        for o in range(self.n_orders):
            # if this is an active order, increase the waiting time
            if self.o_status[o] >= 1:
                self.o_time[o] += 1

        # Check if any order expires
        reward_no_penalty = self.reward
        self.info["RewardNoPenalty"] = reward_no_penalty
        for o in range(self.n_orders):
            if self.o_time[o] >= self.order_promise:
                # Incur the cost to the driver who had accepted the order
                if self.o_status[o] >= 2:
                    # Give order miss penalty and take rewards back that were given during accept and pickup
                    self.reward = (
                        self.reward
                        - self.order_miss_penalty
                        - self.reward_per_order[o] * (self.o_status[o] == 2) / 3
                        - self.reward_per_order[o] * (self.o_status[o] == 3) * 2 / 3
                    )

                    self.late_penalty += self.order_miss_penalty
                    if self.o_status[o] == 3:
                        self.dr_used_capacity -= 1
                self.__reset_order(o)

        # Check if any open order is taken by some other driver
        for o in range(self.n_orders):
            if self.o_status[o] == 1 and np.random.random(1)[0] < self.order_timeout_prob:
                self.__reset_order(o)

        # Create new orders
        for o in range(self.n_orders):
            if self.o_status[o] == 0:
                # Flip a coin to create an order
                if np.random.random(1)[0] < self.order_prob:
                    # Choose a zone
                    zone = np.random.choice(self.num_zones, p=self.order_probs_per_zone)
                    o_x, o_y, from_rest, order_reward = self.__receive_order(zone)
                    self.o_status[o] = 1
                    self.o_time[o] = 0
                    self.o_res_map[o] = from_rest
                    self.o_x[o] = o_x
                    self.o_y[o] = o_y
                    self.reward_per_order[o] = order_reward

    def __reset_order(self, order_num):
        self.o_status[order_num] = 0
        self.o_time[order_num] = 0
        self.o_res_map[order_num] = -1
        self.o_x[order_num] = 0
        self.o_y[order_num] = 0
        self.reward_per_order[order_num] = 0

    def __place_restaurants(self):
        self.res_coordinates = []
        self.res_x = []
        self.res_y = []
        i = 0
        while len(self.res_coordinates) < self.n_restaurants:
            res_x = np.random.choice([i for i in self.map_range_x], 1)[0]
            res_y = np.random.choice([i for i in self.map_range_y], 1)[0]
            res_loc = (res_x, res_y)
            if res_loc not in self.res_coordinates:
                self.res_coordinates.append(res_loc)
                self.res_x.append(res_x)
                self.res_y.append(res_y)
            elif i > 1000:
                print("Something is wrong with the restaurant placement.")
                break

    def __place_driver(self):
        self.dr_x = np.random.choice([i for i in self.map_range_x], 1)[0]
        self.dr_y = np.random.choice([i for i in self.map_range_y], 1)[0]

    def __receive_order(self, zone):
        i = 0  # prevent infinite loop
        while True:
            order_x = np.random.choice([i for i in self.zone_range_x[zone]], 1)[0]
            order_y = np.random.choice([i for i in self.map_range_y], 1)[0]
            # Make sure the order does not overlap with any restaurants
            if (order_x, order_y) not in self.res_coordinates:
                break
            elif i > 1000:
                print("Something is wrong with the restaurant/order locations.")
                break
            else:
                i += 1
        # Determine the restaurant to assign the order
        from_res = np.random.choice([i for i in range(self.n_restaurants)], 1)[0]
        reward = truncnorm.rvs(
            (self.order_reward_min[zone] - self.order_reward_min[zone])
            / self.half_norm_scale_reward_per_zone[zone],
            (self.order_reward_max[zone] - self.order_reward_min[zone])
            / self.half_norm_scale_reward_per_zone[zone],
            self.order_reward_min[zone],
            self.half_norm_scale_reward_per_zone[zone],
            1,
        )[0]
        return order_x, order_y, from_res, reward

    def __reset_state(self):
        assert self.dr_used_capacity == self.o_status.count(3)
        assert self.dr_used_capacity <= self.driver_capacity

        return {
            "action_mask": np.array([1.0] * self.action_space.n),
            "real_obs": np.array(
                self.res_x
                + self.res_y
                + [self.dr_x]
                + [self.dr_y]
                + [self.dr_used_capacity]
                + [self.driver_capacity]
                + self.o_x
                + self.o_y
                + self.o_status
                + self.o_res_map
                + self.o_time
                + self.reward_per_order
            ),
        }

    def __create_state(self):

        assert self.dr_used_capacity == self.o_status.count(3)
        assert self.dr_used_capacity <= self.driver_capacity

        return np.array(
            self.res_x
            + self.res_y
            + [self.dr_x]
            + [self.dr_y]
            + [self.dr_used_capacity]
            + [self.driver_capacity]
            + self.o_x
            + self.o_y
            + self.o_status
            + self.o_res_map
            + self.o_time
            + self.reward_per_order
        )
