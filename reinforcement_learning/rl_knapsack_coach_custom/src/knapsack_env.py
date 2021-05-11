import csv
import queue

import gym
import numpy as np
from capacity import Capacity
from gym import spaces
from item import Item
from knapsack_baseline import (
    get_knapsack_benchmark_sol_hard_greedy_heuristic,
    get_knapsack_solution_medium,
    get_knapsack_solution_simple,
)
from knapsack_view_2D import KnapsackView2D


class KnapSackEnv(gym.Env):
    def render(self, mode="human"):

        if self.knapsack_view is None:
            self.knapsack_view = KnapsackView2D(
                bag_weight_capacity=self.bag_weight_capacity, max_item_value=self.max_item_value
            )

        return self.knapsack_view.update(
            mode=mode,
            selected_item_queue=self.selected_item_queue,
            reward=self.total_reward,
            item=self.item,
            bag_weight=self.bag_weight,
            bag_value=self.bag_value,
        )

    def __init__(self):

        self.min_weight_capacity = Capacity.min_weight
        self.max_weight_capacity = Capacity.max_weight
        self.max_item_value = 100
        self.drop_penalty = -10
        self.time_horizon = 20

        self.bag_weight = None
        self.bag_weight_capacity = None
        self.bag_value = None
        self.time_remaining = None
        self.item = None
        self.items_list = []
        self.selected_item_queue = None

        self.total_reward = 0
        self.episode_count = 0
        self.knapsack_view = None

        self.csv_file = "knapsack_easy.csv"

        # Note: can collapse capacity and sum of weights into one dimension and can remove sum of values
        # state: capacity,
        #        sum of weights of items,
        #        sum of values of items,
        #        item weight,
        #        item value,
        #        time remaining,
        self.observation_space = spaces.Box(
            low=np.array([self.min_weight_capacity, 0, 0, 0, 0, 0]),
            high=np.array(
                [
                    self.max_weight_capacity,  # bag capacity
                    self.max_weight_capacity,  # sum of weights in bag
                    np.inf,  # sum of values in bag
                    self.max_weight_capacity,  # item weight
                    self.max_item_value,  # item value
                    self.time_horizon,  # time remaining
                ]
            ),
            dtype=np.uint32,
        )
        # actions: 0 -> don't put the item in the bag,
        #          1 -> put the item in the bag,
        self.action_space = spaces.Discrete(2)

    def reset(self):

        if self.episode_count == 0:
            self.create_baseline_csv()
        self.bag_weight = 0
        self.bag_value = 0
        self.time_remaining = self.time_horizon
        self.bag_weight_capacity = np.random.randint(
            self.min_weight_capacity, self.max_weight_capacity
        )
        self.item = Item.get_random_item(
            max_weight=Capacity.max_weight, max_value=self.max_item_value
        )
        self.items_list = [self.item]
        initial_state = [
            self.bag_weight_capacity,
            self.bag_weight,
            self.bag_value,
            self.item.weight,
            self.item.value,
            self.time_remaining,
        ]
        self.selected_item_queue = queue.Queue()
        self.total_reward = 0
        self.episode_count += 1

        self.knapsack_view = None

        return initial_state

    def step(self, action):

        done = False
        if action == Action.THROW:  # don't put item in bag
            reward = self.drop_penalty
        elif action == Action.PUT:  # put the item in bag
            if self.bag_weight_capacity < self.bag_weight + self.item.weight:
                # drop the item
                reward = self.drop_penalty
            else:
                self.selected_item_queue.put(self.item)
                self.bag_weight += self.item.weight
                self.bag_value += self.item.value
                reward = self.item.value
        else:
            raise ValueError("Invalid action {}".format(action))

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        if done:
            # get baseline results
            weights = [x.weight for x in self.items_list]
            values = [x.value for x in self.items_list]
            result = get_knapsack_solution_simple(
                weights, values, self.bag_weight_capacity, self.drop_penalty
            )
            print("Baseline reward: ", result[0], "RL Reward: ", self.total_reward)
            # Save it to file
            with open(self.csv_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_count, self.total_reward, result[0]])

        self.item = Item.get_random_item(
            max_weight=Capacity.max_weight, max_value=self.max_item_value
        )
        self.items_list += [self.item]

        state = [
            self.bag_weight_capacity,
            self.bag_weight,
            self.bag_value,
            self.item.weight,
            self.item.value,
            self.time_remaining,
        ]
        info = {}
        return state, reward, done, info

    def create_baseline_csv(self):
        header = ["Episode", "RL Reward", "Baseline Reward"]
        with open(self.csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)


class KnapSackMediumEnv(KnapSackEnv):
    def render(self, mode="human"):

        if self.knapsack_view is None:
            self.knapsack_view = KnapsackView2D(
                bag_weight_capacity=self.bag_weight_capacity,
                max_item_value=self.max_item_value,
                bag_volume_capacity=self.bag_volume_capacity,
            )

        return self.knapsack_view.update(
            mode=mode,
            selected_item_queue=self.selected_item_queue,
            reward=self.total_reward,
            item=self.item,
            bag_weight=self.bag_weight,
            bag_volume=self.bag_volume,
            bag_value=self.bag_value,
        )

    def __init__(self):

        super().__init__()
        self.bag_volume = None
        self.bag_volume_capacity = None
        self.time_horizon = 50

        self.csv_file = "knapsack_medium.csv"

        # Note: can collapse capacity and sum of weights into one dimension and can remove sum of values
        # state: weight_capacity,
        #        volume_capacity,
        #        sum of volumes of items,
        #        sum of weights of items,
        #        sum of values of items,
        #        item weight,
        #        item volume,
        #        item value,
        #        time remaining,
        self.observation_space = spaces.Box(
            low=np.array([Capacity.min_weight, Capacity.min_volume, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array(
                [
                    Capacity.max_weight,  # bag weight capacity
                    Capacity.max_volume,  # bag volume capacity
                    Capacity.max_weight,  # sum of weights in bag
                    Capacity.max_volume,  # sum of volume in bag
                    np.inf,  # sum of values in bag
                    Capacity.max_weight,  # item weight
                    Capacity.max_volume,  # item volume
                    self.max_item_value,  # item value
                    self.time_horizon,  # time remaining
                ]
            ),
        )

    def reset(self):

        if self.episode_count == 0:
            self.create_baseline_csv()
        self.bag_weight = 0
        self.bag_volume = 0
        self.bag_value = 0
        self.time_remaining = self.time_horizon
        self.bag_weight_capacity = np.random.randint(Capacity.min_weight, Capacity.max_weight)
        self.bag_volume_capacity = np.random.randint(Capacity.min_volume, Capacity.max_volume)
        self.item = Item.get_random_item(
            max_value=self.max_item_value,
            max_weight=Capacity.max_weight,
            max_volume=Capacity.max_volume,
        )
        self.items_list = [self.item]
        initial_state = [
            self.bag_weight_capacity,
            self.bag_volume_capacity,
            self.bag_weight,
            self.bag_volume,
            self.bag_value,
            self.item.weight,
            self.item.volume,
            self.item.value,
            self.time_remaining,
        ]
        self.selected_item_queue = queue.Queue()
        self.total_reward = 0
        self.episode_count += 1

        self.knapsack_view = None

        return initial_state

    def step(self, action):

        done = False
        if action == Action.THROW:  # don't put item in bag
            reward = self.drop_penalty
        elif action == Action.PUT:  # put the item in bag
            if (self.bag_weight_capacity < self.bag_weight + self.item.weight) or (
                self.bag_volume_capacity < self.bag_volume + self.item.volume
            ):
                # drop the item
                reward = self.drop_penalty
            else:
                self.selected_item_queue.put(self.item)
                self.bag_weight += self.item.weight
                self.bag_volume += self.item.volume
                self.bag_value += self.item.value
                reward = self.item.value
        else:
            raise ValueError("Invalid action {}".format(action))

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        if done:
            # get baseline results
            if self.episode_count % 50 == 0:
                weights = [x.weight for x in self.items_list]
                volumes = [x.volume for x in self.items_list]
                values = [x.value for x in self.items_list]
                result = get_knapsack_solution_medium(
                    weights,
                    volumes,
                    values,
                    self.bag_weight_capacity,
                    self.bag_volume_capacity,
                    self.drop_penalty,
                )

                print("Baseline reward: ", result[0], "RL Reward: ", self.total_reward)
                # Save it to file
                with open(self.csv_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_count, self.total_reward, result[0]])

        self.item = Item.get_random_item(
            max_value=self.max_item_value,
            max_weight=Capacity.max_weight,
            max_volume=Capacity.max_volume,
        )
        self.items_list += [self.item]
        state = [
            self.bag_weight_capacity,
            self.bag_volume_capacity,
            self.bag_weight,
            self.bag_volume,
            self.bag_value,
            self.item.weight,
            self.item.volume,
            self.item.value,
            self.time_remaining,
        ]
        info = {}
        return state, reward, done, info


class KnapSackHardEnv(KnapSackMediumEnv):
    # items disappear from bag after item_stay_duration time steps

    def __init__(self):

        super().__init__()
        self.time_horizon = 100
        self.item_stay_duration = 5  # time steps

        self.csv_file = "knapsack_hard.csv"

    def reset(self):
        initial_state = super().reset()
        return initial_state

    def step(self, action):
        dummy_item = Item(weight=0, volume=0, value=0)
        done = False
        if action == Action.THROW:  # don't put item in bag
            reward = self.drop_penalty
            self.selected_item_queue.put(dummy_item)
        elif action == Action.PUT:  # put the item in bag
            if (self.bag_weight_capacity < self.bag_weight + self.item.weight) or (
                self.bag_volume_capacity < self.bag_volume + self.item.volume
            ):
                # drop the item
                reward = self.drop_penalty
                self.selected_item_queue.put(dummy_item)
            else:
                self.selected_item_queue.put(self.item)
                self.bag_weight += self.item.weight
                self.bag_volume += self.item.volume
                self.bag_value += self.item.value
                reward = self.item.value
        else:
            raise ValueError("Invalid action {}".format(action))

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        if self.time_remaining < (self.time_horizon - self.item_stay_duration):
            # items disappear after stay duration
            exit_item = self.selected_item_queue.get()
            self.bag_weight -= exit_item.weight
            self.bag_volume -= exit_item.volume

        if done:
            # get baseline results
            if self.episode_count % 10 == 0:
                weights = [x.weight for x in self.items_list]
                volumes = [x.volume for x in self.items_list]
                values = [x.value for x in self.items_list]
                result = get_knapsack_benchmark_sol_hard_greedy_heuristic(
                    weights,
                    volumes,
                    values,
                    self.bag_weight_capacity,
                    self.bag_volume_capacity,
                    self.drop_penalty,
                    self.item_stay_duration,
                )

                print("Baseline reward: ", result[0], "RL Reward: ", self.total_reward)
                # Save it to file
                with open(self.csv_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_count, self.total_reward, result[0]])

        self.item = Item.get_random_item(
            max_value=self.max_item_value,
            max_weight=Capacity.max_weight,
            max_volume=Capacity.max_volume,
        )
        self.items_list += [self.item]
        state = [
            self.bag_weight_capacity,
            self.bag_volume_capacity,
            self.bag_weight,
            self.bag_volume,
            self.bag_value,
            self.item.weight,
            self.item.volume,
            self.item.value,
            self.time_remaining,
        ]
        info = {}
        return state, reward, done, info


class KnapSackCommonEnv(KnapSackMediumEnv):
    # items disappear from bag after item_stay_duration time steps

    def __init__(self):

        super().__init__()
        self.time_horizon = 100
        self.item_stay_duration = 5  # time steps

        self.csv_file = "knapsack_hard.csv"

    def reset(self):
        initial_state = super().reset()
        return initial_state

    def step(self, action):
        dummy_item = Item(weight=0, volume=0, value=0)
        done = False
        if action == Action.THROW:  # don't put item in bag
            reward = self.drop_penalty
            self.selected_item_queue.put(dummy_item)
        elif action == Action.PUT:  # put the item in bag
            if (self.bag_weight_capacity < self.bag_weight + self.item.weight) or (
                self.bag_volume_capacity < self.bag_volume + self.item.volume
            ):
                # drop the item
                reward = self.drop_penalty
                self.selected_item_queue.put(dummy_item)
            else:
                self.selected_item_queue.put(self.item)
                self.bag_weight += self.item.weight
                self.bag_volume += self.item.volume
                self.bag_value += self.item.value
                reward = self.item.value
        else:
            raise ValueError("Invalid action {}".format(action))

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        if self.time_remaining < (self.time_horizon - self.item_stay_duration):
            # items disappear after stay duration
            exit_item = self.selected_item_queue.get()
            self.bag_weight -= exit_item.weight
            self.bag_volume -= exit_item.volume

        if done:
            # get baseline results
            if self.episode_count % 10 == 0:
                weights = [x.weight for x in self.items_list]
                volumes = [x.volume for x in self.items_list]
                values = [x.value for x in self.items_list]
                result = get_knapsack_benchmark_sol_hard_greedy_heuristic(
                    weights,
                    volumes,
                    values,
                    self.bag_weight_capacity,
                    self.bag_volume_capacity,
                    self.drop_penalty,
                    self.item_stay_duration,
                )

                print("Baseline reward: ", result[0], "RL Reward: ", self.total_reward)
                # Save it to file
                with open(self.csv_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_count, self.total_reward, result[0]])

        self.item = Item.get_random_item(
            max_value=self.max_item_value,
            max_weight=Capacity.max_weight,
            max_volume=Capacity.max_volume,
        )
        self.items_list += [self.item]
        state = [
            self.bag_weight_capacity,
            self.bag_volume_capacity,
            self.bag_weight,
            self.bag_volume,
            self.bag_value,
            self.item.weight,
            self.item.volume,
            self.item.value,
            self.time_remaining,
        ]
        info = {}
        return state, reward, done, info


class Action:
    THROW = 0
    PUT = 1
