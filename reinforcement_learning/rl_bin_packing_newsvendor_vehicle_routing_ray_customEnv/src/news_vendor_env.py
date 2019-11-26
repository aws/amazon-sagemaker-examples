from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from gym import spaces

"""
STATE:
For each level
l - 1: on-hand inventory + all in transit inventory
l: price
l+1: cost
l+2: holding cost
l+3: penalty for lost sale (civ)
l+4: mean demand (assuming Poisson distribution)

ACTION:
Order up to level i
"""


class NewsVendorGymEnvironment(gym.Env):

    def render(self, mode='human'):
        pass

    def __init__(self, config={}):
        self.l = config.get("lead time", 5)
        self.gamma = config.get("discount factor", 1)
        self.max_level = 4000
        self.max_action = 2000
        self.step_count = 0
        self.max_steps = 40

        self.max_value = 100.
        self.max_holding_cost = 5.
        self.max_loss_goodwill = 10.
        self.max_mean = 200

        self.inv_dim = max(self.l, 1)
        space_low = self.inv_dim * [0]
        space_high = self.inv_dim * [self.max_level]
        space_low += 5 * [0]
        space_high += [self.max_value, self.max_value, self.max_holding_cost, self.max_loss_goodwill, self.max_mean]

        self.observation_space = spaces.Box(low=np.array(space_low),
                                            high=np.array(space_high), dtype=np.float32)

        # Create action space
        # action[0]: Order up to level
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([self.max_action]), dtype=np.float32)
        self.state = None
        self.reset()

    def reset(self):
        self.step_count = 0

        price = np.random.rand() * self.max_value
        cost = np.random.rand() * price
        holding_cost = np.random.rand() * min(cost, self.max_holding_cost)
        loss_goodwill = np.random.rand() * self.max_loss_goodwill
        mean_demand = np.random.rand() * self.max_mean

        self.state = np.zeros(self.inv_dim + 5)
        self.state[self.inv_dim] = price
        self.state[self.inv_dim + 1] = cost
        self.state[self.inv_dim + 2] = holding_cost
        self.state[self.inv_dim + 3] = loss_goodwill
        self.state[self.inv_dim + 4] = mean_demand

        return self.state

    def break_state(self):
        inv_state = self.state[:self.inv_dim]
        p = self.state[self.inv_dim]
        c = self.state[self.inv_dim + 1]
        h = self.state[self.inv_dim + 2]
        k = self.state[self.inv_dim + 3]
        mu = self.state[self.inv_dim + 4]
        return inv_state, p, c, h, k, mu

    def step(self, action):
        done = False
        inv_state, p, c, h, k, mu = self.break_state()

        buys = max(0, min(action[0], max(0, self.max_level - np.sum(inv_state))))

        demand_realization = np.random.poisson(mu)
        # Compute Reward
        on_hand = inv_state[0]
        if self.l == 0:
            on_hand += buys
        sales = min(on_hand, demand_realization)
        sales_revenue = p * sales
        overage = max(0, on_hand - demand_realization)
        underage = max(0, demand_realization - on_hand)
        #        purchase_cost = c * buys
        purchase_cost = self.gamma ** self.l * c * buys
        holding = overage * h
        penalty_lost_sale = k * underage
        reward = sales_revenue - purchase_cost - holding - penalty_lost_sale

        new_state = np.copy(self.state)
        buys = max(0, min(self.max_level - on_hand, buys))
        if self.l > 1:
            new_state[:self.inv_dim - 1] = np.copy(self.state[1:self.inv_dim])
            new_state[self.l - 1] = buys
            new_state[0] += overage
        elif self.l == 1:
            new_state[0] = overage + buys
        else:
            new_state[0] = overage

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        # reward = reward/100.0 #reduce rewards to smaller values
        self.state = np.copy(new_state)
        info = {'demand realization': demand_realization, 'sales': sales, 'underage': underage, 'overage': overage}
        return new_state, reward, done, info


class NewsVendorGymEnvironmentNormalized(NewsVendorGymEnvironment):
    def __init__(self, config={}):
        super().__init__(config)

        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)

    def step(self, action):
        action = np.clip(action[0], 0, 1)
        action = action * self.max_action
        #         print('Scalar de-normalized env action ', action)
        return super().step(np.array([action]))


class NewsVendorGymEnvironmentDiscrete(NewsVendorGymEnvironment):
    def __init__(self, config={}):
        super().__init__(config)
        #         print("Max actions: ", self.max_action)
        self.action_space = spaces.Discrete(self.max_action)

    def step(self, action):
        action = [action]
        return super().step(action)


if __name__ == "__main__":
    env = NewsVendorGymEnvironmentDiscrete()
    initial_state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        print("Action: ", action)
        state, reward, done, info = env.step(action)
        total_reward += reward
    env.close()
    print('Total reward: ', total_reward)
