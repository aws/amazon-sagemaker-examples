import gym
import numpy as np
from gym import spaces

"""
STATE:
Number of bags at each level
Item size

ACTION:
Choose bag
"""

BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10


class BinPackingGymEnvironment(gym.Env):
    def __init__(self, env_config={}):

        config_defaults = {
            "bag_capacity": 9,
            "item_sizes": [2, 3],
            "item_probabilities": [0.8, 0.2],  # linear waste -> SS: -150 to -340
            # 'item_probabilities': [0.75, 0.25], # perfect pack -> SS: -20 to -100
            # 'item_probabilities': [0.5, 0.5], #bounded waste ->  SS: -11 to -20
            "time_horizon": 1000,
        }

        for key, val in config_defaults.items():
            val = env_config.get(key, val)  # Override defaults with constructor parameters
            self.__dict__[key] = val  # Creates variables like self.plot_boxes, self.save_files, etc
            if key not in env_config:
                env_config[key] = val
        print("Using bin size: ", self.bag_capacity)
        print(
            "Using items sizes {} \n With item probabilities {}".format(
                self.item_sizes, self.item_probabilities
            )
        )
        self.csv_file = "/opt/ml/output/intermediate/binpacking.csv"

        self.episode_count = 0

        # state: number of bags at each level, item size,
        self.observation_space = spaces.Box(
            low=np.array([0] * self.bag_capacity + [0]),
            high=np.array([self.time_horizon] * self.bag_capacity + [max(self.item_sizes)]),
            dtype=np.uint32,
        )

        # actions: select a bag from the different levels possible
        self.action_space = spaces.Discrete(self.bag_capacity)

    def reset(self):
        self.time_remaining = self.time_horizon
        self.item_size = self.__get_item()
        self.num_full_bags = 0

        # an array of size bag capacity that keeps track of
        # number of bags at each level
        self.num_bins_levels = [0] * self.bag_capacity

        initial_state = self.num_bins_levels + [self.item_size]
        self.total_reward = 0
        self.waste = 0
        self.episode_count += 1
        self.bin_type_distribution_map = (
            {}
        )  # level to bin types, to the # of bins for each bin type.
        self.step_count = 0
        return initial_state

    def step(self, action):
        done = False
        self.step_count += 1
        if action >= self.bag_capacity:
            print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            reward = BIG_NEG_REWARD - self.waste
            done = True
        elif action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste = sum of empty spaces in all bags
            self.waste = self.bag_capacity - self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # can't insert item because bin of this level doesn't exist
            print("cannot insert item because bin of this level does not exist")
            reward = BIG_NEG_REWARD - self.waste
            done = True
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            # waste = empty space in the bag
            self.waste = -self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(action)
            if self.num_bins_levels[action] < 0:
                print(self.num_bins_levels[action])
            self.num_bins_levels[action] -= 1

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self.__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map

        return state, reward, done, info

    def __get_item(self):
        num_items = len(self.item_sizes)
        item_index = np.random.choice(num_items, p=self.item_probabilities)
        return self.item_sizes[item_index]

    def __update_bin_type_distribution_map(self, target_bin_util):
        if target_bin_util < 0 or target_bin_util + self.item_size > self.bag_capacity:
            print("Error: Invalid Bin Utilization/Item Size")
            return
        elif target_bin_util > 0 and target_bin_util not in self.bin_type_distribution_map:
            print(
                "Error: bin_type_distribution_map does not contain "
                + str(target_bin_util)
                + " as key!"
            )
            return
        elif (
            target_bin_util > 0
            and target_bin_util in self.bin_type_distribution_map
            and len(self.bin_type_distribution_map[target_bin_util]) == 0
        ):
            print(
                "Error: bin_type_distribution_map has no element at level "
                + str(target_bin_util)
                + " !"
            )
            return
        elif target_bin_util == 0:  # opening a new bin
            if self.item_size not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[self.item_size] = {str(self.item_size): 1}
            elif str(self.item_size) not in self.bin_type_distribution_map[self.item_size]:
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] = 1
            else:
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] += 1
        else:
            key = np.random.choice(list(self.bin_type_distribution_map[target_bin_util].keys()))
            if self.bin_type_distribution_map[target_bin_util][key] <= 0:
                print("Error: Invalid bin count!")
                return
            elif self.bin_type_distribution_map[target_bin_util][key] == 1:
                del self.bin_type_distribution_map[target_bin_util][key]
            else:
                self.bin_type_distribution_map[target_bin_util][key] -= 1

            new_key = self.__update_key_for_bin_type_distribution_map(key, self.item_size)
            if (target_bin_util + self.item_size) not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[target_bin_util + self.item_size] = {new_key: 1}
            elif new_key not in self.bin_type_distribution_map[target_bin_util + self.item_size]:
                self.bin_type_distribution_map[target_bin_util + self.item_size][new_key] = 1
            else:
                self.bin_type_distribution_map[target_bin_util + self.item_size][new_key] += 1

    @staticmethod
    def __update_key_for_bin_type_distribution_map(key, item_size):
        parts = key.split(" ")
        parts.append(str(item_size))
        parts.sort()
        return " ".join(parts)

    def render(self, mode="human", close=False):
        pass


class BinPackingIncrementalWasteGymEnvironment(BinPackingGymEnvironment):
    def step(self, action):
        done = False
        if action >= self.bag_capacity:
            print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            reward = BIG_NEG_REWARD - self.waste
        elif action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste = sum of empty spaces in all bags
            self.waste = self.bag_capacity - self.item_size
            # reward is negative waste
            reward = -1 * self.waste
            self.__update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # can't insert item because bin of this level doesn't exist
            print("cannot insert item because bin of this level does not exist")
            reward = BIG_NEG_REWARD - self.waste
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.__update_bin_type_distribution_map(action)
            self.num_bins_levels[action] -= 1
            # waste = sum of empty spaces in all bags
            self.waste = -self.item_size
            # reward is negative waste
            reward = -1 * self.waste

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self.__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        return state, reward, done, info


class BinPackingNearActionGymEnvironment(BinPackingGymEnvironment):
    def step(self, action):
        done = False
        invalid_action = not (self.__is_action_valid(action))
        if invalid_action:
            action = self.__get_nearest_valid_action(action)

        reward = self.__insert_item(action)

        self.total_reward += reward

        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        # get the next item
        self.item_size = self._BinPackingGymEnvironment__get_item()
        # state is the number of bins at each level and the item size
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        return state, reward, done, info

    def __insert_item(self, action):
        if action == 0:  # new bag
            self.num_bins_levels[self.item_size] += 1
            # waste added by putting in new item
            self.waste = self.bag_capacity - self.item_size
        else:  # insert in existing bag
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.num_bins_levels[action] -= 1
            # waste reduces as we insert item in existing bag
            self.waste = -self.item_size
        # reward is negative waste
        reward = -1 * self.waste
        self._BinPackingGymEnvironment__update_bin_type_distribution_map(action)
        return reward

    def __get_nearest_valid_action(self, action):
        num_actions = self.bag_capacity
        # valid actions have
        valid_actions = list()
        for x in range(1, num_actions):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        if valid_actions:
            # get nearest valid action
            valid_action = min(valid_actions, key=lambda x: abs(x - action))
        else:
            valid_action = 0  # open new bag

        return valid_action

    def __is_action_valid(self, action):
        if action >= self.bag_capacity:
            print("Error: Invalid Action ", action)
            raise
        elif action > (self.bag_capacity - self.item_size):
            # can't insert item because bin overflow
            print("cannot insert item because bin overflow")
            return False
        elif action == 0:  # new bag
            return True
        elif self.num_bins_levels[action] == 0:
            print("cannot insert item because bin of this level does not exist")
            return False
        else:  # insert in existing bag
            return True


class BinPackingContinuousActionEnv(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        # actions: select a bag from the different levels possible
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

    def step(self, action):
        # clip to 0 and 1
        action = np.clip(action, 0, 1)
        # de-normalize to bin size and make it integer
        action = int(action * (self.bag_capacity - 1))
        return super().step(action)


class BinPackingActionMaskGymEnvironment(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        self.observation_space = spaces.Dict(
            {
                # a mask of valid actions (e.g., [0, 0, 1, 0, 0, 1] for 6 max avail)
                "action_mask": spaces.Box(0, 1, shape=(self.action_space.n,), dtype=np.float32),
                "real_obs": self.observation_space,
            }
        )

    def reset(self):
        state = super().reset()
        valid_actions = self.__get_valid_actions()
        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs

    def step(self, action):
        state, rew, done, info = super().step(action)
        valid_actions = self.__get_valid_actions()
        self.action_mask = [1 if x in valid_actions else 0 for x in range(self.action_space.n)]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, rew, done, info

    def __get_valid_actions(self):
        valid_actions = list()
        # get bin levels for which bins exist and item will fit
        for x in range(1, self.action_space.n):
            if self.num_bins_levels[x] > 0:
                if x <= (self.bag_capacity - self.item_size):
                    valid_actions.append(x)
        valid_actions.append(0)  # open new bag
        return valid_actions


if __name__ == "__main__":
    env_config = {
        "bag_capacity": 100,
        "item_sizes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "item_probabilities": [0, 0, 0, 1 / 3, 0, 0, 0, 0, 2 / 3],  # linear waste
        # 'item_probabilities': [0.14, 0.10, 0.06, 0.13, 0.11, 0.13, 0.03, 0.11, 0.19], #bounded waste
        # 'item_probabilities': [0.06, 0.11, 0.11, 0.22, 0, 0.11, 0.06, 0, 0.33], #perfect pack
        "time_horizon": 10000,
    }

    env = BinPackingActionMaskGymEnvironment(env_config)
    initial_state = env.reset()
    done = False
    count = 0
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        print("Action: {0}, Reward: {1:.1f}, Done: {2}".format(action, reward, done))
    print("Total Reward: ", total_reward)
