# Offline RL Training the Cart-pole Model with SageMaker RL and Ray Rllib


Offline reinforcement learning enables the RL agent to learn from offline datasets that have been previously collected. Offline RL is useful in scenarios where 
simulating the actual environment is expensive. Here, we demonstrate how to train an offline RL agent for the familiar Cartpole balancing problem using SageMaker RL
and Ray Rllib. We generate the experiences that make up the offline dataset and then train the agent using IMPALA, an offpolicy RL algorithm.

## Contents
* `rl_cartpole_balancing_using_offline_RL.ipynb`: Main notebook used for training the offline RL agent.
* `source/`  
  * `train-rl-cartpole-ray-gen.py`: Config file for generating the cartpole experiences dataset using PPO algorithm.
  * `train-rl-cartpole-ray-offline-IMPALA.py`: Config file for training the offline RL cartpole agent using IMPALA algorithm.
  * `evaluate-ray.py`: Python script to enable evaluation of the agent performance metrics.
