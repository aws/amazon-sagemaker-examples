import itertools


def tsp_action_go_from_a_to_b(a, b):
    # 0: Up, 1: Down, 2: Left, 3: Right

    action = None
    cur_x = a[0]
    cur_y = a[1]
    tar_x = b[0]
    tar_y = b[1]

    x_diff = tar_x - cur_x
    y_diff = tar_y - cur_y

    if abs(x_diff) >= abs(y_diff):
        # Move horizontally
        if x_diff > 0:
            action = 3
        elif x_diff < 0:
            action = 2
    else:
        # Move vertically
        if y_diff > 0:
            action = 0
        elif y_diff < 0:
            action = 1

    return action


import numpy as np


def create_dist_matrix(all_xy, num_stops):
    # D[i,j] is the cost of going from i to j
    D = {i: {} for i in range(num_stops)}  # index 0 is the restaurant

    # Create distance matrix
    for i in range(num_stops):
        for j in range(i + 1, num_stops):
            dist = manhattan_dist(all_xy[i][0], all_xy[i][1], all_xy[j][0], all_xy[j][1])
            D[i][j] = dist
            D[j][i] = dist

    return D


def tsp_dp_approx_sol(res_xy, orders_xy):
    # This baseline is for the TSP problem,
    # a single agent traveling all orders starting and finishing at a single restaurant
    # assuming res_xy = (res_x, res_y), orders_xy = [(order1_x, order1_y), ...]

    all_xy = [res_xy] + orders_xy
    num_stops = len(all_xy)
    D = create_dist_matrix(all_xy, num_stops)

    # Best cost in stage i for each order
    DP = {i: {} for i in range(num_stops)}
    # Subsequent visits in the best route from stage i on for each order
    DP_will_visit = {i: {} for i in range(num_stops)}

    # DP solution, backwards
    for i in reversed(range(num_stops)):
        # This is the final visit to the restaurant
        if i == num_stops - 1:
            for o in range(1, num_stops):
                DP[i][o] = D[o][0]
                DP_will_visit[i][o] = [o]
        else:
            if i == 0:
                stop_list = [0]
            else:
                stop_list = range(1, num_stops)

            for o in stop_list:
                min_dist = np.inf
                min_o_next = None
                for o_next in range(1, num_stops):
                    if o_next in DP_will_visit[i + 1].keys():
                        if o not in DP_will_visit[i + 1][o_next]:
                            cost = D[o][o_next] + DP[i + 1][o_next]
                            if cost < min_dist:
                                min_o_next = o_next
                                min_dist = cost

                if min_o_next:
                    DP[i][o] = min_dist
                    DP_will_visit[i][o] = [o] + DP_will_visit[i + 1][min_o_next]

    print(DP)
    print(DP_will_visit)
    return DP[0], DP_will_visit[0][0] + [0]


def manhattan_dist(x1, y1, x2, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)


def tsp_dp_opt_sol(res_xy, orders_xy):
    all_xy = [res_xy] + orders_xy
    num_stops = len(all_xy)
    D = create_dist_matrix(all_xy, num_stops)
    C = {}  # Subtour cost dictionary, (set of nodes in the subtour, last node)
    P = {}  # Subtour path dictionary, (set of nodes in the subtour, last node)

    # Initialize C
    for o in range(1, num_stops):
        C[frozenset({o}), o] = D[0][o]
        P[frozenset({o}), o] = [0, o]

    for s in range(2, num_stops):
        for S in itertools.combinations(range(1, num_stops), s):
            for o in S:
                search_keys = [(frozenset(S) - {o}, m) for m in S if m != o]
                search_list = [C[S_o, m] + D[m][o] for S_o, m in search_keys]
                min_val = min(search_list)
                opt_key = search_keys[search_list.index(min_val)]
                C[frozenset(S), o] = min_val
                P[frozenset(S), o] = P[opt_key] + [o]

    final_set = frozenset(range(1, num_stops))
    search_list = [C[final_set, o] + D[o][0] for o in final_set]
    best_cost = min(search_list)
    opt_final_order = search_list.index(best_cost) + 1
    best_route = P[final_set, opt_final_order] + [0]

    return best_cost, best_route
