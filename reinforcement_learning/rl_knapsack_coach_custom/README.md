# Solving Knapsack problem with Amazon SageMaker RL

This shows an example of how to use SageMaker RL to address a canonical operations research problem. We choose which items to put in the Knapsack. Our objective is to maximize the value of the items in the bag; but we cannot put all the items in as the bag capacity is limited.

## Contents

* `rl_knapsack_clippedppo_coach_tensorflow_customEnv.ipynb`: Notebook used for training  the policy to address the knapsack problem.
* `src/`
  * `knapsack_env.py`: custom environments and simulator defined here.
  * `train-coach.py`: launcher for coach training.
  * `evaluate-coach.py`: launcher for coach evaluation.
  * `preset-knapsack-clippedppo.py`: coach preset for Clipped PPO.

