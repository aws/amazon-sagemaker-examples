#  Unity3D Game with Amazon SageMaker RL

This folder contains examples of how to use RL to train an agent to play Unity3D game using Amazon SageMaker Reinforcement Learning. Customer can choose using [example environment](https://github.com/Unity-Technologies/ml-agents/blob/742c2fbf01188fbf27e82d5a7d9b5fd42f0de67a/docs/Learning-Environment-Examples.md) provided by Unity Toolkit or bring their own customized Unity executables.


## Contents

* `rl_unity_ray.ipynb`: notebook for training an RL agent.


* `src/`
  * `train-unity.py`: Entrypoint file to starting a training job
  * `evaluate-unity.py`: Entrypoint file to starting a evaluation job 