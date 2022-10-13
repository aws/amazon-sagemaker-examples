######################
Reinforcement Learning
######################

.. image:: rl_mountain_car_coach_gymEnv/successful_policy.gif
  :width: 600
  :alt: mountain car policy

Get started with RL
===================

.. toctree::
   :maxdepth: 0

   rl_mountain_car_coach_gymEnv/rl_mountain_car_coach_gymEnv
   rl_cartpole_ray/rl_cartpole_ray_gymEnv
   rl_roboschool_ray/rl_roboschool_ray

..
   TODO: fix model deployment headings


Cart pole
=========

A cart pole simulation is the act of balancing a broom upright by balancing it on your hand.
The broom is the "pole" and your hand is replaced with a "cart" moving back and forth on a linear track.
This simplified example works in 2 dimensions, so the cart can only move in a line back and forth, and the pole can only fall forwards or backwards, not to the sides.
These examples use PyTorch or TensorFlow and SageMaker RL to solve a cart pole problem.

.. toctree::
   :maxdepth: 0

   rl_cartpole_coach/rl_cartpole_coach_gymEnv
   rl_cartpole_batch_coach/rl_cartpole_batch_coach
   rl_managed_spot_cartpole_coach/rl_managed_spot_cartpole_coach_gymEnv


Contextual bandits
==================

Explore a number of actions with contextual bandits algorithms in SageMaker.

.. toctree::
   :maxdepth: 0

   bandits_recsys_movielens_testbed/bandits_movielens_testbed
   bandits_statlog_vw_customEnv/bandits_statlog_vw_customEnv


Roboschool
===========

`Roboschool <https://github.com/openai/roboschool/tree/master/roboschool>`_ is a physics simulator that is commonly used to train RL policies for robotic systems.

.. toctree::
   :maxdepth: 0

   rl_roboschool_ray/rl_roboschool_ray_automatic_model_tuning
   rl_roboschool_ray/rl_roboschool_ray_distributed
   rl_roboschool_stable_baselines/rl_roboschool_stable_baselines


Use cases
=========

Autoscaling
-----------

This example demonstrates how to use RL to address scaling a production service by adding and removing resources (e.g. servers or EC2 instances) in reaction to a dynamic load.

.. toctree::
   :maxdepth: 0

   rl_predictive_autoscaling_coach_customEnv/rl_predictive_autoscaling_coach_customEnv


Energy
------

Training an RL algorithm in a real HVAC system can take time to converge as well as potentially lead to hazardous settings as the agent explores its state space.
This example uses the `EnergyPlus <https://energyplus.net/>`_ simulator to showcase how you can train an HVAC optimization RL model with Amazon SageMaker.

.. toctree::
   :maxdepth: 0

   rl_hvac_coach_energyplus/rl_hvac_coach_energyplus


Game play
---------

Use RL to train an agent to play in a Unity3D environment.

.. toctree::
   :maxdepth: 0

   rl_unity_ray/rl_unity_ray


Game server
-----------

A reinforcement learning-based system using SageMaker Autopilot and SageMaker RL that learns to allocate resources in response to player usage patterns.

.. toctree::
   :maxdepth: 0

   rl_game_server_autopilot/sagemaker/rl_gamerserver_ray


Knapsack problem
----------------

Use SageMaker RL to address a canonical operations research problem, aka, a knapsack problem.

.. toctree::
   :maxdepth: 0

   rl_knapsack_coach_custom/rl_knapsack_coach_customEnv


Object tracker
--------------

Use RL to train a TurtleBot object tracker using Amazon SageMaker Reinforcement Learning and AWS RoboMaker.

.. toctree::
   :maxdepth: 0

   rl_objecttracker_robomaker_coach_gazebo/rl_objecttracker_coach_robomaker


Network compression
-------------------

Network to network compression via policy gradient reinforcement learning.

.. toctree::
   :maxdepth: 0

   rl_network_compression_ray_custom/rl_network_compression_ray_custom


Portfolio management
--------------------

Use SageMaker RL to manage a stock portfolio by continuously reallocating several stocks.

.. toctree::
   :maxdepth: 0

   rl_portfolio_management_coach_customEnv/rl_portfolio_management_coach_customEnv


Resource allocation
-------------------

Solve resource allocation problems with SageMaker RL.

.. toctree::
   :maxdepth: 0

   rl_resource_allocation_ray_customEnv/rl_bin_packing_ray_custom
   rl_resource_allocation_ray_customEnv/rl_news_vendor_ray_custom
   rl_resource_allocation_ray_customEnv/rl_vehicle_routing_problem_ray_custom


Tic-tac-toe
------------

Play global thermonuclear war with a computer.

.. toctree::
   :maxdepth: 0

   rl_tic_tac_toe_coach_customEnv/rl_tic_tac_toe_coach_customEnv


Stock Trading
------------

Try stock trading with SageMaker RL.

.. toctree::
   :maxdepth: 0

   rl_stock_trading_coach_customEnv/rl_stock_trading_coach_customEnv

 
Traveling salesman problem
--------------------------

Use SageMaker RL to solve this classic problem with a twist: a restaurant delivery service on a 2D gridworld.

.. toctree::
   :maxdepth: 0

   rl_traveling_salesman_vehicle_routing_coach/rl_traveling_salesman_vehicle_routing_coach
