import copy
from time import sleep

import numpy as np
from VRP_baseline_utils import decide_action
from VRP_env import VRPEasyEnv, VRPMediumEnv


def get_mean_baseline_reward(env=VRPEasyEnv(), num_of_episodes=100):
    env_total_rewards = []
    for i in range(num_of_episodes):
        env.reset()
        done = False
        init = True
        visit_stops = []
        total_reward = 0
        while not done:
            if init:
                prev_o_status = [-1] * env.n_orders
                init = False
            action, visit_stops = decide_action(prev_o_status, env, visit_stops)
            prev_o_status = copy.deepcopy(env.o_status)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward

        # print("Total reward: ", total_reward)
        env_total_rewards += [total_reward]

    return np.mean(env_total_rewards), np.std(env_total_rewards)


if __name__ == "__main__":
    env = VRPEasyEnv()
    env.reset()
    # env.render()
    done = False
    init = True
    visit_stops = []
    total_reward = 0
    while not done:
        if init:
            prev_o_status = [-1] * env.n_orders
            init = False
        action, visit_stops = decide_action(prev_o_status, env, visit_stops)
        # print(env.o_status)
        # print("Visit", visit_stops)
        prev_o_status = copy.deepcopy(env.o_status)
        next_state, reward, done, _ = env.step(action)
        # print("Action: {0}, Reward: {1:.1f}, Done: {2}"
        #      .format(action, reward, done))

        total_reward += reward
        # env.render()
        # sleep(0.1)
        # if done:
        #    sleep(2)

    print("Total reward: ", total_reward)
