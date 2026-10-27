# Contextual Bandits with Amazon SageMaker RL

This Notebook demonstrates how you can manage your own contextual multi-armed bandit workflow on SageMaker using the built-in Vowpal Wabbit (VW) container to train and deploy contextual bandit models. We show how to train these models that interact with a live environment (using a simulated client application) and continuously update the model with efficient exploration.


## Contents

- `bandits_statlog_vw_custom.ipynb`: Notebook used for running the contextual bandit notebook.<br>
- `config.yaml`: The configuration parameters used for the bandit example.<br>
- `sim_app`: Simulated client application that pings SageMaker for recommended action given a state. Also computes the rewards for each interaction.<br>
- `common`: Code that manages the different AWS components required for training workflow.<br>
- `src`:
    - `train-vw.py`: Script for training with Vowpal Wabbit library.
    - `eval-cfa-vw.py`: Script for evaluation with Vowpal Wabbit library.
