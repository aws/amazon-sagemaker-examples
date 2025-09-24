import copy
import itertools

import numpy as np


def decide_action(prev_o_status, env, visit_stops):
    action = 0
    o_x = env.o_x
    o_y = env.o_y
    dr_x = env.dr_x
    dr_y = env.dr_y
    o_status = env.o_status

    driver_xy = (dr_x, dr_y)
    order_xy = list(zip(o_x, o_y))
    if prev_o_status == o_status:
        # Nothing has changed, move to the next stop if available
        if visit_stops:
            if driver_xy == visit_stops[0] and len(visit_stops) > 1:
                visit_stops = visit_stops[1:]
                action = vrp_action_go_from_a_to_b(driver_xy, visit_stops[0])
            else:
                action = vrp_action_go_from_a_to_b(driver_xy, visit_stops[0])
    else:
        # Naively accept any outstanding order
        if 1 in o_status:
            order_to_accept = o_status.index(1)
            action = 5 + order_to_accept
        else:
            new_orders = []
            delivered_expired_orders = []
            recently_accepted_orders = []
            for oi, os in enumerate(o_status):
                if os != prev_o_status[oi]:
                    # Order status has changed
                    if os == 0:
                        delivered_expired_orders.append(oi)
                    elif os == 1:
                        new_orders.append(oi)
                    elif os == 2:
                        recently_accepted_orders.append(oi)
            # Replan if there are new orders
            if new_orders:
                visit_stops = extract_state_for_dp(env)
            # Replan for accepted orders
            elif recently_accepted_orders:
                visit_stops = extract_state_for_dp(env)
            # No new orders, but some orders delivered/expired, remove the stop
            elif delivered_expired_orders:
                delivered_expired_xy = [order_xy[i] for i in delivered_expired_orders]
                visit_stops = [xy for xy in visit_stops if xy not in delivered_expired_xy]
            if visit_stops:
                action = vrp_action_go_from_a_to_b(driver_xy, visit_stops[0])

    return action, visit_stops


def vrp_action_go_from_a_to_b(a, b):
    # 0: Up, 1: Down, 2: Left, 3: Right
    action = 0
    cur_x = a[0]
    cur_y = a[1]
    tar_x = b[0]
    tar_y = b[1]

    x_diff = tar_x - cur_x
    y_diff = tar_y - cur_y

    if abs(x_diff) >= abs(y_diff):
        # Move horizontally
        if x_diff > 0:
            action = 4
        elif x_diff < 0:
            action = 3
    else:
        # Move vertically
        if y_diff > 0:
            action = 1
        elif y_diff < 0:
            action = 2

    return action


def extract_state_for_dp(env_view):
    # print("Solving the VRP problem.")
    driver_xy = (env_view.dr_x, env_view.dr_y)
    res_xy = env_view.res_coordinates
    order_xy = list(zip(env_view.o_x, env_view.o_y))
    picked_up_xy = [o for i, o in enumerate(order_xy) if env_view.o_status[i] == 3]
    res_o = [
        [
            r_xy,
            [
                o
                for i, o in enumerate(order_xy)
                if env_view.o_status[i] == 2 and env_view.o_res_map[i] == ri
            ],
        ]
        for ri, r_xy in enumerate(res_xy)
    ]

    sdict = {"driver_loc": driver_xy, "picked_up": picked_up_xy, "res_o": res_o}

    s = State(sdict=sdict, CAP=env_view.driver_capacity)
    s.get_cost_to_go()
    return s.opt_next


class State(object):
    def __init__(self, sdict={}, state=None, DP_TREE={}, CAP=np.inf):
        self.sdict = sdict
        self.x = sdict["driver_loc"][0]
        self.y = sdict["driver_loc"][1]
        self.opt_next = []
        self.opt_cost_to_go = None
        if not state:
            self.state = self._get_hashable_state(self.sdict)
        else:
            self.state = state

        self.to_nodes = set()
        self._populate_to_nodes(DP_TREE, CAP)

    def get_cost_to_go(self):
        if self.opt_cost_to_go is not None:
            return self.opt_cost_to_go

        elif len(self.to_nodes) == 0:
            self.opt_cost_to_go = 0
            return self.opt_cost_to_go

        else:
            best_cost = np.inf
            best_next = None
            for next in self.to_nodes:
                cost_to_next = abs(self.x - next.x) + abs(self.y - next.y)
                next_opt_cost = next.get_cost_to_go()
                candidate_cost = cost_to_next + next_opt_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_next = next

            self.opt_next = [(best_next.x, best_next.y)] + best_next.opt_next
            self.opt_cost_to_go = best_cost

            return self.opt_cost_to_go

    def _get_hashable_state(self, sdict):
        driver_loc = sdict["driver_loc"]  # (x,y)
        picked_up_xy = sdict["picked_up"]  # {(x1, y1), (x2, y2), ...}
        res_o = sdict["res_o"]  # [ [(res1_x, res1_y), {(x11, y11), (x12, y12), ...} ],
        #   [(res2_x, res2_y), {(x21, y21), (x22, y22), ...} ],
        #   ...
        # ]
        if picked_up_xy:
            state = (driver_loc,) + (tuple(sorted(picked_up_xy)),)
        else:
            state = (driver_loc, ())
        hashable_res_vec = ()
        for res_vec in res_o:
            res_xy = res_vec[0]  # (x,y)
            orders = tuple(sorted(res_vec[1]))
            res_vec_h = ((res_xy, orders),)
            hashable_res_vec += res_vec_h

        state += hashable_res_vec
        return state

    def _populate_to_nodes(self, DP_TREE, CAP):
        # Option 1: Deliver one of the orders that have been picked up
        for order_to_deliver in self.sdict["picked_up"]:
            sdict_new = copy.deepcopy(self.sdict)
            sdict_new["driver_loc"] = order_to_deliver
            sdict_new["picked_up"] = set(self.sdict["picked_up"]) - {order_to_deliver}
            new_state = self._get_hashable_state(sdict_new)
            if new_state in DP_TREE:
                # print('State in the tree already', new_state)
                self.to_nodes.add(DP_TREE[new_state])
            else:
                # print('Creating a node for the state', new_state)
                new_node = State(sdict=sdict_new, state=new_state, DP_TREE=DP_TREE, CAP=CAP)
                self.to_nodes.add(new_node)
                DP_TREE[new_state] = new_node

        available_capacity = CAP - len(self.sdict["picked_up"])

        if available_capacity > 0:
            # Option 2: Visit one of the states to pick up orders
            for i, res_to_visit in enumerate(self.sdict["res_o"]):
                # Check if there is any orders to pick up
                res_i_xy = res_to_visit[0]
                res_i_orders = res_to_visit[1]
                if len(res_i_orders) > 0:
                    sdict_new = copy.deepcopy(self.sdict)
                    sdict_new["driver_loc"] = res_i_xy
                    n = min(
                        available_capacity, len(res_i_orders)
                    )  # Fill to the capacity at the restaurant
                    # Do a greedy search for the orders to be picked
                    orders_to_be_picked = set()
                    orders_remaining = res_i_orders
                    last_loc = res_i_xy
                    while len(orders_to_be_picked) < n:
                        closest_dist = np.inf
                        closest_order = None
                        for on in orders_remaining:
                            dist = abs(last_loc[0] - on[0]) + abs(last_loc[1] - on[1])
                            if dist < closest_dist:
                                closest_order = on
                        if closest_order is not None:
                            orders_to_be_picked.add(closest_order)
                        else:
                            break
                        last_loc = closest_order
                        orders_remaining = list(set(orders_remaining) - {closest_order})

                    sdict_new["picked_up"] = list(set(sdict_new["picked_up"]) | orders_to_be_picked)
                    sdict_new["res_o"][i] = [res_i_xy, orders_remaining]
                    if (
                        self.sdict["driver_loc"] != sdict_new["driver_loc"]
                    ):  # Prevent visiting the same restaurant with no other stops in between
                        new_state = self._get_hashable_state(sdict_new)
                        if new_state in DP_TREE:
                            # print('State in the tree already', new_state)
                            self.to_nodes.add(DP_TREE[new_state])
                        else:
                            # print('Creating a node for the state', new_state)
                            new_node = State(
                                sdict=sdict_new, state=new_state, DP_TREE=DP_TREE, CAP=CAP
                            )
                            self.to_nodes.add(new_node)
                            DP_TREE[new_state] = new_node

        # if len(DP_TREE)%100==0:
        #    print("Creating the DP states:", len(DP_TREE))
