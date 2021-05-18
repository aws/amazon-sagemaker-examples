import queue
import random

import numpy as np


def get_knapsack_solution_simple(weights, values, c_max, penalty, debug=False):
    if len(weights) != len(values):
        print("values and weights vectors do not have the same length")
        return

    T = len(weights)
    reward = [[0 for x in range(0, c_max + 1)] for y in range(1, T + 1)]
    # note: reward[i][c_w] has the reward in period (i+1) with residual capacity c_w

    for i in range(T, 0, -1):
        for j in range(0, c_max + 1):
            if i == T:
                if j < weights[i - 1]:
                    reward[i - 1][j] = -penalty
                else:
                    reward[i - 1][j] = values[i - 1]
            else:
                if j < weights[i - 1]:
                    reward[i - 1][j] = -penalty + reward[i][j]
                else:
                    reward[i - 1][j] = max(
                        -penalty + reward[i][j], values[i - 1] + reward[i][j - weights[i - 1]]
                    )

    knapsack_value = reward[0][c_max]
    optimal_packing = find_sol_simple(1, c_max, reward, weights, values, penalty)

    if debug:
        test_sol_valid_simple(knapsack_value, optimal_packing, weights, values, c_max, penalty)

    return [knapsack_value, optimal_packing]


def find_sol_simple(i, c_w, reward, weights, values, penalty):
    T = len(weights)
    if i == T:
        if reward[i - 1][c_w] == -penalty:
            return [0]
        else:
            return [1]
    else:
        if reward[i - 1][c_w] == -penalty + reward[i][c_w]:
            return [0] + find_sol_simple(i + 1, c_w, reward, weights, values, penalty)
        else:
            return [1] + find_sol_simple(
                i + 1, c_w - weights[i - 1], reward, weights, values, penalty
            )


def test_sol_valid_simple(knapsack_value, optimal_packing, weights, values, c_max, penalty):
    if (
        np.dot(optimal_packing, values) - penalty * (len(optimal_packing) - np.sum(optimal_packing))
    ) != knapsack_value:
        print("total solution value does not match up to knapsack value.")
    elif np.dot(optimal_packing, weights) > c_max:
        print("solution violates knapsack capacity.")
    else:
        print("passed all tests!")


def get_knapsack_solution_medium(
    weights, volumes, values, c_weight_max, c_vol_max, penalty, debug=False
):
    if (len(weights) != len(values)) or (len(volumes) != len(weights)):
        print("values, volumes, weights vectors do not have the same length")
        return

    T = len(weights)
    reward = [
        [[0 for z in range(0, c_vol_max + 1)] for x in range(0, c_weight_max + 1)]
        for y in range(1, T + 1)
    ]
    # note: reward[i][c_w] has the reward in period (i+1) with residual capacity c_w

    for i in range(T, 0, -1):
        for j in range(0, c_weight_max + 1):
            for z in range(0, c_vol_max + 1):
                if i == T:
                    if j < weights[i - 1] or z < volumes[i - 1]:
                        reward[i - 1][j][z] = -penalty
                    else:
                        reward[i - 1][j][z] = values[i - 1]
                else:
                    if j < weights[i - 1] or z < volumes[i - 1]:
                        reward[i - 1][j][z] = -penalty + reward[i][j][z]
                    else:
                        reward[i - 1][j][z] = max(
                            -penalty + reward[i][j][z],
                            values[i - 1] + reward[i][j - weights[i - 1]][z - volumes[i - 1]],
                        )

    knapsack_value = reward[0][c_weight_max][c_vol_max]
    optimal_packing = find_sol_medium(
        1, c_weight_max, c_vol_max, reward, volumes, weights, values, penalty
    )

    if debug:
        test_sol_valid_medium(
            knapsack_value,
            optimal_packing,
            weights,
            volumes,
            values,
            c_weight_max,
            c_vol_max,
            penalty,
        )

    return [knapsack_value, optimal_packing]


def find_sol_medium(i, c_w, c_v, reward, volumes, weights, values, penalty):
    T = len(weights)
    if i == T:
        if reward[i - 1][c_w][c_v] == -penalty:
            return [0]
        else:
            return [1]
    else:
        if reward[i - 1][c_w][c_v] == -penalty + reward[i][c_w][c_v]:
            return [0] + find_sol_medium(i + 1, c_w, c_v, reward, volumes, weights, values, penalty)
        else:
            return [1] + find_sol_medium(
                i + 1,
                c_w - weights[i - 1],
                c_v - volumes[i - 1],
                reward,
                volumes,
                weights,
                values,
                penalty,
            )


def test_sol_valid_medium(
    knapsack_value, optimal_packing, weights, volumes, values, c_weight_max, c_vol_max, penalty
):
    if (
        np.dot(optimal_packing, values) - penalty * (len(optimal_packing) - np.sum(optimal_packing))
    ) != knapsack_value:
        print("total solution value does not match up to knapsack value.")
    elif np.dot(optimal_packing, weights) > c_weight_max:
        print("solution violates knapsack weight capacity.")
    elif np.dot(optimal_packing, volumes) > c_vol_max:
        print("solution violates knapsack volume capacity.")
    else:
        print("passed all tests!")


def get_knapsack_solution_hard(
    weights, volumes, values, c_weight_max, c_vol_max, penalty, duration, debug=False
):
    out_of_boundary_penalty = -10000
    if (len(weights) != len(values)) or (len(volumes) != len(weights)):
        print("values, volumes, weights vectors do not have the same length")
        return

    T = len(weights)
    reward = [
        [[{} for z in range(0, c_vol_max + 1)] for x in range(0, c_weight_max + 1)]
        for y in range(1, T + 1)
    ]
    # note: reward[i][c_w] has the reward in period (i+1) with residual capacity c_w

    for i in range(T, 0, -1):
        # print(i)
        s = "{0:0" + str(min(duration, i - 1)) + "b}"
        for j in range(0, c_weight_max + 1):
            for z in range(0, c_vol_max + 1):
                past_action_to_reward_map = reward[i - 1][j][z]
                for w in range(0, 2 ** (min(duration, i - 1))):
                    key = s.format(w)
                    # if z == 0 and j == 0:
                    #    print(key)
                    if (
                        i > duration
                    ):  # in this case an item from duration = d periods ago leaves the knapsack
                        j_hat = j + int(key[0]) * weights[i - duration - 1]
                        z_hat = z + int(key[0]) * volumes[i - duration - 1]
                        # checking to make sure state is valid
                        if j_hat > c_weight_max or z_hat > c_vol_max:
                            past_action_to_reward_map[key] = out_of_boundary_penalty
                            continue
                    else:
                        j_hat = j
                        z_hat = z

                    if i == T:
                        if i > duration:
                            if j_hat < weights[i - 1] or z_hat < volumes[i - 1]:
                                past_action_to_reward_map[key] = -penalty
                            else:
                                past_action_to_reward_map[key] = values[i - 1]
                        else:
                            if j < weights[i - 1] or z < volumes[i - 1]:
                                past_action_to_reward_map[key] = -penalty
                            else:
                                past_action_to_reward_map[key] = values[i - 1]
                    else:
                        if i > duration:
                            new_key1 = key[1 : len(key)] + str(0)
                            new_key2 = key[1 : len(key)] + str(1)
                            if j_hat < weights[i - 1] or z_hat < volumes[i - 1]:
                                past_action_to_reward_map[key] = (
                                    -penalty + reward[i][j_hat][z_hat][new_key1]
                                )
                            else:
                                past_action_to_reward_map[key] = max(
                                    -penalty + reward[i][j_hat][z_hat][new_key1],
                                    values[i - 1]
                                    + reward[i][j_hat - weights[i - 1]][z_hat - volumes[i - 1]][
                                        new_key2
                                    ],
                                )
                        elif i > 1:
                            if j < weights[i - 1] or z < volumes[i - 1]:
                                past_action_to_reward_map[key] = (
                                    -penalty + reward[i][j][z][key + str(0)]
                                )
                            else:
                                past_action_to_reward_map[key] = max(
                                    -penalty + reward[i][j][z][key + str(0)],
                                    values[i - 1]
                                    + reward[i][j - weights[i - 1]][z - volumes[i - 1]][
                                        key + str(1)
                                    ],
                                )
                        else:
                            if j < weights[i - 1] or z < volumes[i - 1]:
                                past_action_to_reward_map[key] = -penalty + reward[i][j][z][str(0)]
                            else:
                                past_action_to_reward_map[key] = max(
                                    -penalty + reward[i][j][z][str(0)],
                                    values[i - 1]
                                    + reward[i][j - weights[i - 1]][z - volumes[i - 1]][str(1)],
                                )

    knapsack_value = reward[0][c_weight_max][c_vol_max]["0"]
    optimal_packing = find_sol_hard(
        i, c_weight_max, c_vol_max, "0", reward, weights, volumes, values, penalty, duration
    )

    if debug:
        test_sol_valid_hard(
            knapsack_value,
            optimal_packing,
            weights,
            volumes,
            values,
            c_weight_max,
            c_vol_max,
            penalty,
            duration,
        )

    return [knapsack_value, optimal_packing]


def find_sol_hard(i, c_w, c_v, key, reward, weights, volumes, values, penalty, duration):
    T = len(weights)
    if i == T:
        if reward[i - 1][c_w][c_v][key] == -penalty:
            return [0]
        else:
            return [1]
    else:
        if i > duration:  # in this case an item from duration = d periods ago leaves the knapsack
            j_hat = c_w + int(key[0]) * weights[i - duration - 1]
            z_hat = c_v + int(key[0]) * volumes[i - duration - 1]
            if j_hat > c_weight_max or z_hat > c_vol_max:
                return [-1]  # out of bound error
        else:
            j_hat = c_w
            z_hat = c_v

        past_action_to_reward_map = reward[i - 1][c_w][c_v]

        # print(past_action_to_reward_map)
        if i > duration:
            new_key1 = key[1 : len(key)] + str(0)
            new_key2 = key[1 : len(key)] + str(1)

            if past_action_to_reward_map[key] == -penalty + reward[i][j_hat][z_hat][new_key1]:
                return [0] + find_sol_hard(
                    i + 1,
                    j_hat,
                    z_hat,
                    new_key1,
                    reward,
                    weights,
                    volumes,
                    values,
                    penalty,
                    duration,
                )
            else:
                return [1] + find_sol_hard(
                    i + 1,
                    j_hat - weights[i - 1],
                    z_hat - volumes[i - 1],
                    new_key2,
                    reward,
                    weights,
                    volumes,
                    values,
                    penalty,
                    duration,
                )
        elif i > 1:
            if past_action_to_reward_map[key] == -penalty + reward[i][c_w][c_v][key + str(0)]:
                return [0] + find_sol_hard(
                    i + 1,
                    c_w,
                    c_v,
                    key + str(0),
                    reward,
                    weights,
                    volumes,
                    values,
                    penalty,
                    duration,
                )
            else:
                return [1] + find_sol_hard(
                    i + 1,
                    c_w - weights[i - 1],
                    c_v - volumes[i - 1],
                    key + str(1),
                    reward,
                    weights,
                    volumes,
                    values,
                    penalty,
                    duration,
                )
        else:
            if past_action_to_reward_map[key] == -penalty + reward[i][c_w][c_v][str(0)]:
                return [0] + find_sol_hard(
                    i + 1, c_w, c_v, str(0), reward, weights, volumes, values, penalty, duration
                )
            else:
                return [1] + find_sol_hard(
                    i + 1,
                    c_w - weights[i - 1],
                    c_v - volumes[i - 1],
                    str(1),
                    reward,
                    weights,
                    volumes,
                    values,
                    penalty,
                    duration,
                )


def test_sol_valid_hard(
    knapsack_value,
    optimal_packing,
    weights,
    volumes,
    values,
    c_weight_max,
    c_vol_max,
    penalty,
    duration,
):
    if (
        np.dot(optimal_packing, values) - penalty * (len(optimal_packing) - np.sum(optimal_packing))
    ) != knapsack_value:
        print("total solution value does not match up to knapsack value.")
        print(
            (
                np.dot(optimal_packing, values)
                - penalty * (len(optimal_packing) - np.sum(optimal_packing))
            )
        )
        print(knapsack_value)
        return

    if duration >= len(weights):
        if np.dot(optimal_packing, weights) > c_weight_max:
            print("solution violates knapsack weight capacity.")
        elif np.dot(optimal_packing, volumes) > c_vol_max:
            print("solution violates knapsack volume capacity.")
        else:
            print("passed all tests!")
    else:
        for i in range(0, len(weights) - duration + 1):
            l = []
            for j in range(0, i):
                l.append(0)
            for j in range(i, i + duration):
                l.append(1)
            for j in range(i + duration, len(weights)):
                l.append(0)
            if np.dot(np.multiply(optimal_packing, l), weights) > c_weight_max:
                print("solution violates knapsack weight capacity for the subset")
                print(np.multiply(optimal_packing, l))
                return
            elif np.dot(np.multiply(optimal_packing, l), volumes) > c_vol_max:
                print("solution violates knapsack volume capacity for the subset")
                rint(np.multiply(optimal_packing, l))
                return
        else:
            print("passed all tests!")


def get_knapsack_benchmark_sol_hard_greedy_heuristic(
    weights, volumes, values, c_weight_max, c_vol_max, penalty, duration, debug=False
):
    if (len(weights) != len(values)) or (len(volumes) != len(weights)):
        print("values, volumes, weights vectors do not have the same length")
        return

    min_item_weight = min(weights)
    min_item_vol = min(volumes)
    max_item_value = max(values)

    max_ratio = 1000
    max_trials = 10
    discount_factor = 2 / 3
    dummy_item = [0, 0, 0]
    objs = []
    sols = []
    if min_item_weight > 0 or min_item_vol > 0:
        max_ratio = min(max_item_value / min_item_weight, max_item_value / min_item_vol)

    for j in range(1, max_trials + 1):
        # print(j)
        threshold = max_ratio * (discount_factor ** j)
        item_queue = queue.Queue()
        obj = 0
        total_weight_in_knapsack = 0
        total_volume_in_knapsack = 0
        sol = []
        for i in range(0, len(weights)):
            if i >= duration:
                exit_item = item_queue.get()
                total_weight_in_knapsack = total_weight_in_knapsack - exit_item[0]
                total_volume_in_knapsack = total_volume_in_knapsack - exit_item[1]

            if (values[i] / weights[i] < threshold) or (values[i] / volumes[i] < threshold):
                item_queue.put(dummy_item)
                obj = obj - penalty
                sol.append(0)
            else:
                if (total_weight_in_knapsack + weights[i] <= c_weight_max) and (
                    total_volume_in_knapsack + volumes[i] <= c_vol_max
                ):
                    total_weight_in_knapsack = total_weight_in_knapsack + weights[i]
                    total_volume_in_knapsack = total_volume_in_knapsack + volumes[i]
                    item_queue.put([weights[i], volumes[i], values[i]])
                    obj = obj + values[i]
                    sol.append(1)
                else:
                    item_queue.put(dummy_item)
                    obj = obj - penalty
                    sol.append(0)
        objs.append(obj)
        sols.append(sol)
    max_index = objs.index(max(objs))
    if debug:
        test_sol_valid_hard(
            knapsack_value,
            optimal_packing,
            weights,
            volumes,
            values,
            c_weight_max,
            c_vol_max,
            penalty,
            duration,
        )

    return [objs[max_index], sols[max_index]]


if __name__ == "__main__":
    debug = False
    weights = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
    values = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
    volumes = []
    for x in range(0, len(weights)):
        volumes.append(random.randint(1, 100))

    c_weight_max = 200
    c_vol_max = 200
    penalty = 10
    duration = 5

    print(
        get_knapsack_benchmark_sol_hard_greedy_heuristic(
            weights, volumes, values, c_weight_max, c_vol_max, penalty, duration
        )
    )
