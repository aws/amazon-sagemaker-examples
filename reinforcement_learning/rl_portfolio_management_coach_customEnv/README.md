# Portfolio Management with Amazon SageMaker RL

Portfolio management is the process of constant redistribution of a capital into a set of different financial assets. Given the historic prices of a list of stocks and current portfolio allocation, the goal is to maximize the return while restraining the risk. In this demo, we use a reinforcement learning solution framework to manage the portfolio by continuously reallocating several stocks. Based on the setup in [1], we use a tensor input constructed from historical price data, then apply an actor-critic policy gradient algorithm to accommodate the continuous actions (reallocations). The customized environment is constructed using Open AI gym, and the RL agents are trained using AWS SageMaker.

[1] Jiang, Zhengyao, Dixing Xu, and Jinjun Liang. "A deep reinforcement learning framework for the financial portfolio management problem." arXiv preprint arXiv:1706.10059 (2017).

## Contents

* `rl_portfolio_management_coach_customEnv.ipynb`: notebook used for training portfolio management policy.
* `src/`
  * `datasets/stocks_history_target.h5`: source data. See notebook for license.
  * `config.py`: configurations including data selection, data directory.
  * `utils.py`: utility functions.
  * `portfolio_env.py`: custom environments and simulator.
  * `train-coach.py`: launcher for coach training.
  * `evaluate-coach.py`: launcher for coach evaluation.
  * `preset-portfolio-management-clippedppo.py`: coach preset for Clipped PPO.


## Risk Disclaimer (for live-trading)

This notebook is for educational purposes only. Past trading performance does not guarantee future performance. The loss in trading can be substantial, and therefore 
**investors should use all trading strategies at their own risk**.
