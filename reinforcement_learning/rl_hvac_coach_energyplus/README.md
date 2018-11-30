# HVAC with Amazon SageMaker RL

## What is HVAC and EnergyPlus?

HVAC stands for Heating, Ventilation and Air Conditioning and is responsible for keeping us warm and comfortable indoors. HVAC takes up a whopping 50% of the energy in a building and accounts for 40% of energy use in the US, and several control system optimizations have been proposed to reduce this energy use while ensuring thermal comfort.

In a modern building, data such as weather, occupancy and equipment use are collected routinely and can be used to optimize HVAC energy use. Reinforcement Learning (RL) is a good fit as it can learn patterns in the data and identify strategies to control the system so as to reduce energy. Several recent research efforts have shown that RL can reduce HVAC energy consumption by 15-20% [1, 2].

As training an RL algorithm in a real HVAC system can take time to converge as well as potentially lead to hazardous settings as the agent explores its state space, we turn to a simulator to train the agent. EnergyPlus (https://energyplus.net/) is an open source, state of the art HVAC simulator from the US Department of Energy. We use a simple example with this simulator to showcase how we can train an RL model easily with Amazon SageMaker.

## Contents

* `rl_hvac_coach_energyplus.ipynb`: notebook used for training HVAC policy.
* `Dockerfile`: used to build custom container that extends MXNet Coach container.
* `src/`
  * `eplus`: directory for EnergyPlus Python interface for environments.
  * `evaluate-baseline.py`: fixed baseline policy for establishing baseline performance.
  * `evaluate-coach.py`: entry point script for evaluating trained model.
  * `preset-energy-plus-clipped-ppo.py`: coach preset for Clipped PPO.
  * `train-coach.py`: entry point script for coach training.

### Container Versions:

Python: 3.6.4
EnergyPlus: 8.8.0 (https://github.com/NREL/EnergyPlus/releases/tag/v8.8.0)
