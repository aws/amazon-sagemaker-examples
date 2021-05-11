"""
Modified from https://github.com/vermouth1992/drl-portfolio-management/blob/master/src/environment/portfolio.py
"""
import csv
from pprint import pprint

import gym
import gym.spaces
import numpy as np
from config import *

from utils import *


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(
        self,
        steps=730,  # 2 years
        trading_cost=0.0025,
        time_cost=0.00,
        window_length=7,
        start_idx=0,
        sample_start_date=None,
    ):
        """
        An environment for financial portfolio management.
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """

        datafile = DATA_DIR
        history, abbreviation = read_stock_history(filepath=datafile)
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx
        self.csv_file = CSV_DIR

        self.src = DataGenerator(
            history=history,
            abbreviation=abbreviation,
            steps=steps,
            window_length=window_length,
            start_idx=start_idx,
            start_date=sample_start_date,
        )
        self.sim = PortfolioSim(
            asset_names=abbreviation, trading_cost=trading_cost, time_cost=time_cost, steps=steps
        )

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=(len(self.src.asset_names) + 1,), dtype=np.float32
        )

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(abbreviation), window_length, history.shape[-1] - 1),
            dtype=np.float32,
        )

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1,))

        # normalise just in case
        weights = np.clip(action, 0, 1)
        weights /= weights.sum() + EPS
        weights[0] += np.clip(
            1 - weights.sum(), 0, 1
        )  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), (
            "all action values should be between 0 and 1. Not %s" % action
        )
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights
        )

        observation, done1, ground_truth_obs = self.src._step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation_concat = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation_concat[:, -1, 3]
        open_price_vector = observation_concat[:, -1, 0]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info["market_value"] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        info["date"] = index_to_date(self.start_idx + self.src.idx + self.src.step)
        info["steps"] = self.src.step
        info["next_obs"] = ground_truth_obs

        self.infos.append(info)

        if done1:
            # Save it to file
            keys = self.infos[0].keys()
            with open(self.csv_file, "w", newline="") as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.infos)
        return observation, reward, done1 or done2, info

    def reset(self):
        return self._reset()

    def _reset(self):
        self.infos = []
        self.sim.reset()
        observation, ground_truth_obs = self.src.reset()
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation_concat = np.concatenate((cash_observation, observation), axis=0)
        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)
        info = {}
        info["next_obs"] = ground_truth_obs
        return observation


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(
        self, history, abbreviation, steps=730, window_length=50, start_idx=0, start_date=None
    ):
        """

        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert history.shape[0] == len(abbreviation), "Number of stock is not consistent"
        import copy

        self.steps = steps + 1
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)

    def _step(self):
        # get price matrix (open, close, high, low) from history.
        self.step += 1
        obs = self.data[:, self.step : self.step + self.window_length, :].copy()

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[
            :, self.step + self.window_length : self.step + self.window_length + 1, :
        ].copy()

        done = self.step >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step = 0
        # get data for this episode, each episode might be different.
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps
            )
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date) - self.start_idx
            assert (
                self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps
            ), "Invalid start date, must be window_length day after start date and simulation steps day before end date"
        data = self._data[:, self.idx - self.window_length : self.idx + self.steps + 1, :4]
        self.data = data
        return (
            self.data[:, self.step : self.step + self.window_length, :].copy(),
            self.data[
                :, self.step + self.window_length : self.step + self.window_length + 1, :
            ].copy(),
        )


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, "w1 and y1 must have the same shape"
        assert y1[0] == 1.0, "y1[0] must be 1"

        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + EPS)  # (eq7) weights evolve into
        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio
        assert mu1 < 1.0, "Cost is larger than current holding"

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        p1 = np.clip(p1, 0, np.inf)  # no shorts

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + EPS) / (p0 + EPS))  # log rate of return
        reward = r1 / self.steps * 1000.0  # (22) average logarithmic accumulated return
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = bool(p1 == 0)

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names))
        self.p0 = 1.0
