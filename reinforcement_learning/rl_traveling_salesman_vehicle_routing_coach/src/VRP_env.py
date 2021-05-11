import gym
import numpy as np
from gym import spaces
from rl_operations_research_baselines.VRP.VRP_view_2D import VRPView2D

""" 

STATE:
Restaurant x
Restaurant y
Driver x
Driver y
Driver used capacity
Driver max capacity
Order status: inactive/delivered/not-created, open, accepted, picked up
Order x
Order y
Order to restaurant map
Time elapsed per order after 'open'

ACTION:
- no action
- L,R,U,D
- accept an order

Easy:
#Restaurants: 1
#Orders: 2
Order Promise: Infinite( = Episode length)
Order Timeout: Infinite( = Episode length)
Driver Capacity: Infinite( =  # Orders)
"""


class VRPEasyEnv(gym.Env):
    def render(self, mode="human", close=False):

        if self.vrp_view is None:
            self.vrp_view = VRPView2D(
                n_restaurants=self.n_restaurants,
                n_orders=self.n_orders,
                map_quad=self.map_quad,
                grid_size=25,
            )
        return self.vrp_view.update(
            res_x=self.res_x,
            res_y=self.res_y,
            o_status=self.o_status,
            o_x=self.o_x,
            o_y=self.o_y,
            dr_x=self.dr_x,
            dr_y=self.dr_y,
            mode=mode,
        )

    def __init__(
        self,
        n_restaurants=1,
        n_orders=2,
        order_prob=0.3,
        driver_capacity=5,
        map_quad=(2, 2),
        order_promise=100,
        order_timeout=100,
        episode_length=100,
    ):

        self.vrp_view = None
        self.map_quad = map_quad

        self.n_orders = n_orders
        self.n_restaurants = n_restaurants
        self.driver_capacity = driver_capacity
        self.order_prob = order_prob
        self.order_promise = order_promise
        self.order_timeout = order_timeout
        self.dr_used_capacity = 0
        self.o_x = []
        self.o_y = []
        self.o_status = []
        self.o_res_map = []
        self.o_time = []

        self.dr_x = None
        self.dr_y = None

        self.game_over = False
        self.state = []
        self.reward = None

        # store the inputs
        self.episode_length = episode_length
        self.clock = 0

        # map boundaries
        self.map_min_x = -map_quad[0]
        self.map_max_x = +map_quad[0]
        self.map_min_y = -map_quad[1]
        self.map_max_y = +map_quad[1]
        self.map_range_x = range(-self.map_max_x, self.map_max_x + 1)
        self.map_range_y = range(-self.map_max_y, self.map_max_y + 1)

        # restaurant x position limits
        res_x_min = [self.map_min_x] * n_restaurants
        res_x_max = [self.map_max_x] * n_restaurants
        # restaurant y position limits
        res_y_min = [self.map_min_y] * n_restaurants
        res_y_max = [self.map_max_y] * n_restaurants

        # driver x position limits
        dr_x_min = [self.map_min_x]
        dr_x_max = [self.map_max_x]
        # driver y position limits
        dr_y_min = [self.map_min_y]
        dr_y_max = [self.map_max_y]

        dr_used_capacity_min = [0]
        dr_used_capacity_max = [driver_capacity]

        # n_orders for x position limits
        o_x_min = [self.map_min_x] * n_orders
        o_x_max = [self.map_max_x] * n_orders
        # n_orders for y position limits
        o_y_min = [self.map_min_y] * n_orders
        o_y_max = [self.map_max_y] * n_orders

        # order status: 0 - inactive(not created, cancelled, delivered), 1 - open, 2 - accepted, 3 - picked-up
        o_status_min = [0] * n_orders
        o_status_max = [3] * n_orders

        # order-restaurant mapping, i.e. which the order belongs to which restaurant
        o_res_map_min = [-1] * n_orders
        o_res_map_max = [n_restaurants - 1] * n_orders

        # time elapsed since the order has been placed
        o_time_min = [0] * n_orders
        o_time_max = [order_timeout] * n_orders

        # Create the observation space
        self.observation_space = spaces.Box(
            low=np.array(
                res_x_min
                + res_y_min
                + dr_x_min
                + dr_y_min
                + dr_used_capacity_min
                + [driver_capacity]
                + o_x_min
                + o_y_min
                + o_status_min
                + o_res_map_min
                + o_time_min
                + [order_promise]
                + [order_timeout]
            ),
            high=np.array(
                res_x_max
                + res_y_max
                + dr_x_max
                + dr_y_max
                + dr_used_capacity_max
                + [driver_capacity]
                + o_x_max
                + o_y_max
                + o_status_max
                + o_res_map_max
                + o_time_max
                + [order_promise]
                + [order_timeout]
            ),
            dtype=np.int16,
        )

        # Action space: no action, up, down, left, right, accept order i
        self.action_space = spaces.Discrete(5 + n_orders)

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

        self.vrp_view = None

        return self.__create_state()

    def step(self, action):

        done = False
        self.reward = 0
        self.__update_driver_parameters(action)
        self.__update_environment_parameters()
        state = self.__create_state()

        # Update the clock
        self.clock += 1
        if self.clock >= self.episode_length:
            done = True

        info = {}
        return state, self.reward, done, info

    def __update_driver_parameters(self, action):
        if action == 0:  # no action
            pass
        elif action == 1:  # UP
            self.dr_y = min(self.map_max_y, self.dr_y + 1)
        elif action == 2:  # DOWN
            self.dr_y = max(self.map_min_y, self.dr_y - 1)
        elif action == 3:  # LEFT
            self.dr_x = max(self.map_min_x, self.dr_x - 1)
        elif action == 4:  # RIGHT
            self.dr_x = min(self.map_max_x, self.dr_x + 1)
        elif action > 4:  # accept order i
            o = action - 5  # order index
            # if order is open and driver has capacity, accept it
            if self.o_status[o] == 1 and self.dr_used_capacity < self.driver_capacity:
                self.o_status[o] = 2
                self.dr_used_capacity += 1

        # Check for pick-ups for each order accepted by the driver but not picked up/delivered yet.
        for r in range(self.n_restaurants):
            res_x = self.res_x[r]
            res_y = self.res_y[r]
            if self.dr_x == res_x and self.dr_y == res_y:
                # The driver is at a restaurant. Check if any accepted order can be picked up from here.
                for o in range(self.n_orders):
                    # if an order is assigned to this driver, if it is open
                    # and if it is ordered from the restaurant the driver is at, then pick it up
                    if self.o_status[o] == 2 and self.o_res_map[o] == r:
                        self.o_status[o] = 3  # set order status to picked up
                        self.reward += (self.order_timeout - self.o_time[o]) * 0.1

        # Check for deliveries
        for o in range(self.n_orders):
            # If order is picked up by driver and driver is at delivery location, deliver the order
            if self.o_status[o] == 3 and (self.dr_x == self.o_x[o] and self.dr_y == self.o_y[o]):
                # 50 cents of tip/penalty for early/late delivery.
                # self.reward += (self.order_promise - self.o_time[o]) * 0.5
                self.reward += (
                    max(0.0, (self.order_promise - self.o_time[o]) * 0.5)
                    + (self.order_timeout - self.o_time[o]) * 0.15
                )
                self.dr_used_capacity -= 1
                self.o_status[o] = 0
                self.o_time[o] = 0
                self.o_res_map[o] = -1
                self.o_x[o] = 0
                self.o_y[o] = 0

    def __update_environment_parameters(self):
        # Update the waiting times
        for o in range(self.n_orders):
            # if this is an active order, increase the waiting time
            if self.o_status[o] > 1:
                self.o_time[o] += 1
        # Check if any order expires
        for o in range(self.n_orders):
            if self.o_time[o] >= self.order_timeout:
                # Incur the cost to the driver who had accepted the order
                # print("Order", o, "has expired.")
                if self.o_status[o] >= 2:
                    self.reward -= self.order_timeout * 0.5
                    self.dr_used_capacity -= 1

                self.o_status[o] = 0
                self.o_time[o] = 0
                self.o_res_map[o] = -1
                self.o_x[o] = 0
                self.o_y[o] = 0
        # Create new orders
        for o in range(self.n_orders):
            if self.o_status[o] == 0:
                # Flip a coin to create an order
                if np.random.random(1)[0] < self.order_prob:
                    o_x, o_y, r = self.__receive_order()
                    self.o_x[o] = o_x
                    self.o_y[o] = o_y
                    self.o_res_map[o] = r
                    self.o_status[o] = 1

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

    def __receive_order(self):
        i = 0  # prevent infinite loop
        while True:
            order_x = np.random.choice([i for i in self.map_range_x], 1)[0]
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
        return order_x, order_y, from_res

    def __create_state(self):
        return (
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
            + [self.order_promise]
            + [self.order_timeout]
        )


"""
Medium:
#Restaurants: 1
#Orders: 10
Order Promise: Infinite( = Episode length)
Order Timeout: Infinite( = Episode length)
Driver Capacity: Infinite( =  # Orders)
"""


class VRPMediumEnv(VRPEasyEnv):
    def __init__(self):
        super().__init__(
            n_restaurants=1,
            n_orders=10,
            order_prob=0.9,
            driver_capacity=4,
            map_quad=(8, 8),
            order_promise=200,
            order_timeout=400,
            episode_length=2000,
        )


"""
Hard:
#Restaurants: 2
#Orders: 10
Order Promise: 30
Order Timeout: 30
Driver Capacity: 3
"""


class VRPHardEnv(VRPEasyEnv):
    def __init__(self):
        super().__init__(
            n_restaurants=2,
            n_orders=3,
            order_prob=0.9,
            driver_capacity=4,
            map_quad=(10, 10),
            order_promise=60,
            order_timeout=120,
            episode_length=5000,
        )
