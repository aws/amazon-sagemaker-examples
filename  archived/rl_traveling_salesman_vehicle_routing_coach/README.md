# Traveling Salesman and Vehicle Routing with Amazon SageMaker RL

The travelling salesman problem (TSP) is a classic algorithmic problem in the field of computer science and operations research. Given a list of cities and the distances between each pair of cities, the problem is to find the shortest possible route that visits each city and returns to the origin city.

The problem is NP-complete as the number of combinations of cities grows larger as we add more cities.

In the classic version of the problem, the salesman picks a city to start, travels through remaining cities and returns to the original city.

In this version, we have slightly modified the problem, presenting it as a restaurant order delivery problem on a 2D gridworld. The agent (driver) starts at the restaurant, a fixed point on the grid. Then, delivery orders appear elsewhere on the grid. The driver needs to visit the orders, and return to the restaurant, to obtain rewards. Rewards are proportional to the time taken to do this (equivalent to the distance, as each timestep moves one square on the grid).

Vehicle Routing is a similar problem where the algorithm optimizes the movement of a fleet of vehicles. In our formulation, we have a delivery driver who accepts orders from customers, picks up food from a restaurant and delivers it to the customer. The driver optimizes to increase the number of successful deliveries within a time limit.

## Contents

* `rl_traveling_salesman_vehicle_routing_coach`: notebook used for training traveling salesman and vehicle routing policies.
* `src/`
  * `TSP_env.py`: traveling salesman problem is defined here.
  * `TSP_view_2D.py`: visualizer for the traveling salesman problem.
  * `TSP_baseline.py`: baseline implementation of traveling salesman.
  * `TSP_baseline_util.py`: helper file for baseline implmentation.
  * `VRP_env.py`: vehicle routing problem is defined here.
  * `VRP_abstract_env.py`: defines an easier version of vehicle routing problem where the driver knows the path to go from one place to another.
  * `VRP_view_2D.py`: visualizer for the vehicle routing problem.
  * `VRP_baseline.py`: baseline implementation of vehicle routing.
  * `VRP_baseline_util.py`: helper file for baseline implmentation.
  * `train-coach.py`: launcher for coach training.
  * `evaluate-coach.py`: launcher for coach evaluation.
  * `preset-tsp-easy.py`: coach preset for Clipped PPO for the easy version of TSP.
  * `preset-tsp-medium.py`: coach preset for Clipped PPO for the medium version of TSP.
  * `preset-vrp-easy.py`: coach preset for Clipped PPO for the easy version of VRP.
