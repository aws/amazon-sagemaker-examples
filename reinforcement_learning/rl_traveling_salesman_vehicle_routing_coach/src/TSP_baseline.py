from time import sleep

import numpy as np
from TSP_baseline_utils import tsp_action_go_from_a_to_b, tsp_dp_approx_sol, tsp_dp_opt_sol
from TSP_env import TSPEasyEnv, TSPMediumEnv


def get_mean_baseline_reward(env=TSPEasyEnv(), num_of_episodes=100):
    env_total_rewards = []
    for i in range(num_of_episodes):
        env.reset()
        optimal_sln = True
        done = False
        tsp_solved = False
        visit_list = []
        action = None
        total_reward = 0
        while not done:
            agt_xy = (env.agt_x, env.agt_y)

            if not tsp_solved:
                stops_xy = [(0, 0)] + [o for o in zip(env.o_x, env.o_y)]
                if optimal_sln:
                    _, visit_order = tsp_dp_opt_sol((0, 0), stops_xy[1:])
                else:
                    _, visit_order = tsp_dp_approx_sol((0, 0), stops_xy[1:])
                visit_list = [stops_xy[v] for v in visit_order]
                tsp_solved = True

            if agt_xy in visit_list:
                visit_list.remove(agt_xy)

            if visit_list:
                action = tsp_action_go_from_a_to_b(agt_xy, visit_list[0])

            if action is None:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            # print("Action: {0}, Reward: {1:.1f}, Done: {2}"
            #  .format(action, reward, done))
        # print("Total reward: ", total_reward)
        env_total_rewards += [total_reward]
    return np.mean(env_total_rewards), np.std(env_total_rewards)


if __name__ == "__main__":
    env = TSPMediumEnv()
    env.reset()
    optimal_sln = True
    done = False
    tsp_solved = False
    visit_list = []
    action = None
    total_reward = 0
    while not done:
        agt_xy = (env.agt_x, env.agt_y)

        if not tsp_solved:
            print("Solving TSP")
            stops_xy = [(0, 0)] + [o for o in zip(env.o_x, env.o_y)]
            if optimal_sln:
                _, visit_order = tsp_dp_opt_sol((0, 0), stops_xy[1:])
            else:
                _, visit_order = tsp_dp_approx_sol((0, 0), stops_xy[1:])
            visit_list = [stops_xy[v] for v in visit_order]
            print("TSP solution", visit_list)
            tsp_solved = True

        if agt_xy in visit_list:
            visit_list.remove(agt_xy)

        if visit_list:
            action = tsp_action_go_from_a_to_b(agt_xy, visit_list[0])

        if action is not None:
            next_state, reward, done, _ = env.step(action)
            env.render()
            print("Action: {0}, Reward: {1:.1f}, Done: {2}".format(action, reward, done))
            total_reward += reward
        else:
            print("No action taken.")

        env.render()
        sleep(0.3)
        if done:
            sleep(2)
    print("Total reward: ", total_reward)
