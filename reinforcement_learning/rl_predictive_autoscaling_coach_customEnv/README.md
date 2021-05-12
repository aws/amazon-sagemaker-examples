# Predictive Auto-scaling with Amazon SageMaker RL

This example demonstrates how to use RL to address a very common problem in production operation of software systems: scaling a production service
by adding and removing resources (e.g. servers or EC2 instances) in reaction to dynamically changing load. This example is a simple toy
demonstrating how one might begin to address this real and challenging problem.  It generates a fake simulated load with daily and weekly
variations and occasional spikes. The simulated system has a delay between when new resources are requested and when they become available
for serving requests. The customized environment is constructed based on Open AI Gym, with 10000 time steps in one episode.
At each time step, the agent is allowed to add machines AND subtract machines.

## Contents

* `rl_predictive_autoscaling_coach_customEnv.ipynb`: notebook used for training predictive auto-scaling policy.
* `src/`
  * `autoscalesim.py`: custom environments and simulator defined here.
  * `gymhelper.py`: generate `gym.space.Box` from custom environments and simulator
  * `train-coach.py`: launcher for coach training.
  * `evaluate-coach.py`: launcher for coach evaluation.
  * `preset-autoscale-ppo.py`: coach preset for Clipped PPO.
  * `preset-autoscale-a3c.py`: coach preset for A3C.
