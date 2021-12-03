Offline reinforcement learning enables the RL agent to learn from an offline data that has been previously collected. Offline RL is useful in scenarios where 
simulating the actual environment is expensive. Here, we demonstrate how to train an offline RL agent for the familiar Cartpole balancing problem using SageMaker RL
and Ray Rllib. We generate the experiences that make up the offline dataset and then train the agent using IMPALA, an offpolicy RL algorithm
