# Stock Trading with Amazon SageMaker RL

This example applies the deep Q-network method to train an agent that will trade a single share to maximize profit. The goal is to demonstrate how to go beyond the Atari games and apply RL to a different practical domain. Based on the setup in chapter 8 of [1], we use one-minute historical share price intervals, and then apply a Double DQN architecture to accommodate a simple set of discrete trading actions: do nothing, buy a single share, and close the position. The customized environment is constructed using Open AI Gym and the RL agents are trained using Amazon SageMaker. 

[1] Maxim Lapan. "[Deep Reinforcement Learning Hands-On." Packt (2018)](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter08/lib/environ.py).

## Contents

* `rl_stock_trading_coach_customEnv.ipynb`: notebook used for training stock trading agent.
* `src/`
  * `datasets/YNDX_160101_161231.csv`: source data. See notebook for license.
  * `config.py`: configurations including data selection, data directory.
  * `data.py`: data functions.
  * `trading_env.py`: custom environments and simulator.
  * `train-coach.py`: launcher for coach training.
  * `evaluate-coach.py`: launcher for coach evaluation.
  * `preset-stock-trading-ddqn.py`: coach preset for Double DQN.

## Risk Disclaimer (for live-trading)

This notebook is for educational purposes only. Past trading performance does not guarantee future performance. The loss in trading can be substantial, and therefore 
**investors should use all trading strategies at their own risk**.
