# Resource Allocation with Amazon SageMaker RL

Resource allocation problems in Operations Research and Computer Science often manifest as online and stochastic decision making tasks. In this domain, we consider three canonical problems and demonstrate how to solve them using RL.

- Bin packing. Given items of different sizes, one needs to pack them into as few bins as possible. 
- News vendor. In inventory management, one needs to decide on an ordering decision (how much of an item to purchase from a supplier) to cover a single period of uncertain demand. 
- Vehicle routing problem. Given one or more vehicles and a set of locations, one needs to find the route that reaches all locations with minimal operational cost. 

## Contents

* `rl_bin_packing_ray_custom.ipynb`: notebook used for training bin packing policy.
* `rl_news_vendor_ray_custom.ipynb`: notebook used for training news vendor policy.
* `rl_vehicle_routing_problem_ray_custom.ipynb`: notebook used for training vehicle routing problem policy.
* `src/`
  * `bin_packing_env.py`: customer environment for bin packing problem.
  * `news_vendor_env.py`: customer environment for news vendor problem.
  * `vrp_env.py`: customer environment for vehicle routing problem.
  * `vrp_view_2D.py`: visualizer for the vehicle routing problem.
  * `train-bin_packing.py`: PPO config for training bin packing problem.
  * `train-news_vendor.py`: PPO config for training news vendor problem.
  * `train-vehicle_routing_problem.py`: APEX-DQN config for training vehicle routing problem.
  * `model.py`: custom model used in Ray to mask invalid actions.
  * `utils.py`: helper function for vehicle routing problem.
* `images/`: images used in the notebooks.
