# Amazon SageMaker Examples

### Common Reinforcement Learning Examples

These examples demonstrate how to train reinforcement learning models on SageMaker for a wide range of applications.

-  [Contextual Bandit with Live Environment](bandits_statlog_vw_customEnv) illustrates how you can manage your own contextual multi-armed bandit workflow on SageMaker using the built-in [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) (VW) container to train and deploy contextual bandit models.
-  [Cartpole](rl_cartpole_coach) uses SageMaker RL base [docker image](https://github.com/aws/sagemaker-rl-container) to balance a broom upright.
-  [Cartpole Batch](rl_cartpole_batch_coach) uses batch RL techniques to train Cartpole with offline data.
-  [Cartpole Spot Training](rl_managed_spot_cartpole_coach) uses SageMaker Managed Spot instances at a lower cost.
-  [DeepRacer](rl_deepracer_robomaker_coach_gazebo) gives a glimse of architecture used to get the DeepRacer working with AWS RoboMaker.
-  [HVAC](rl_hvac_coach_energyplus) optimizes energy use based on the [EnergyPlus](https://energyplus.net/) simulator.
-  [Knapsack](rl_knapsack_coach_custom) is an example of using RL to address operations research problem.
-  [Mountain Car](rl_mountain_car_coach_gymEnv) is a classic control RL problem, in which an under-powered car is tasked with climbing a steep mountain, and is only successful when it reaches the top.
-  [Network Compression](rl_network_compression_ray_custom) reduces the size of a trained network using a RL algorithm.
-  [Object Tracker](rl_objecttracker_robomaker_coach_gazebo) trains a TurtleBot object tracker using Amazon SageMaker RL coupled with AWS RoboMaker.
-  [Portfolio Management](rl_portfolio_management_coach_customEnv) shows how to re-distribute a capital into a set of different financial assets using RL algorithms.
-  [Predictive Auto-scaling](rl_predictive_autoscaling_coach_customEnv) scales a production service via RL approach by adding and removing resources in reaction to dynamically changing load.
-  [Resource Allocation](rl_resource_allocation_ray_customEnv) solves three canonical online and stochastic decision making problems using RL algorithms.
-  [Roboschool Ray](rl_roboschool_ray) demonstrates how to use [Ray](https://rise.cs.berkeley.edu/projects/ray/) to scale RL training in different ways, and how to leverage SageMaker's Automatic Model Tuning functionality to optimize the training of an RL model.
-  [Roboschool Stable Baseline](rl_roboschool_stable_baselines) is an example of using [stable-baselines](https://stable-baselines.readthedocs.io/en/master/) to train RL algorithms.
-  [Tic-tac-toe](rl_tic_tac_toe_coach_customEnv) uses RL to train a policy and then plays locally and interactively within the notebook.
-  [Traveling Salesman and Vehicle Routing](rl_traveling_salesman_vehicle_routing_coach) is an example of using RL to address operations research problems.
-  [Game Server Auto-pilot](rl_game_server_autopilot) Reduce player wait time by autoscaling game-servers deployed in EKS cluster using RL to add and remove EC2 instances as per dynamic player usage.

### FAQ
https://github.com/awslabs/amazon-sagemaker-examples#faq 
