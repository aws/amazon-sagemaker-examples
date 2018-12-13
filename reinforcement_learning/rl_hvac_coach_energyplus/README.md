# HVAC with Amazon SageMaker RL

## What is HVAC and EnergyPlus?

HVAC stands for Heating, Ventilation and Air Conditioning and is responsible for keeping us warm and comfortable indoors. HVAC takes up a whopping 50% of the energy in a building [1, 2] and accounts for 10% of global electricity use [3], and several control system optimizations have been proposed to reduce this energy use while ensuring thermal comfort [4, 5, 6].

In a modern building, data such as weather, occupancy and equipment use are collected routinely and can be used to optimize HVAC energy use. Reinforcement Learning (RL) is a good fit as it can learn patterns in the data and identify strategies to control the system so as to reduce energy. Several recent research efforts have shown that RL can reduce HVAC energy consumption by 15-20% [7, 8].

As training an RL algorithm in a real HVAC system can take time to converge as well as potentially lead to hazardous settings as the agent explores its state space, we turn to a simulator to train the agent. [EnergyPlus](https://energyplus.net/) is an open source, state of the art HVAC simulator from the US Department of Energy. We use a simple example with this simulator to showcase how we can train an RL model easily with Amazon SageMaker.

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

## References

1. Residential Energy Consumption Survey - https://www.eia.gov/consumption/residential/
2. Energy Use in Commercial Buildings - https://www.eia.gov/energyexplained/index.php?page=us\_energy\_commercial  
3. The Future of Cooling - https://www.iea.org/futureofcooling/ 
4. Afram, Abdul, and Farrokh Janabi-Sharifi. "Theory and applications of HVAC control systems–A review of model predictive control (MPC)." Building and Environment 72 (2014): 343-355.
5. Fong, Kwong Fai, Victor Ian Hanby, and Tin-Tai Chow. "System optimization for HVAC energy management using the robust evolutionary algorithm." Applied Thermal Engineering 29, no. 11-12 (2009): 2327-2334.
6. Balaji, Bharathan, Jian Xu, Anthony Nwokafor, Rajesh Gupta, and Yuvraj Agarwal. "Sentinel: Occupancy based HVAC actuation using existing WiFi infrastructure within commercial buildings." In Proceedings of the 11th ACM Conference on Embedded Networked Sensor Systems, p. 17. ACM, 2013.
7. Wei, Tianshu, Yanzhi Wang, and Qi Zhu. "Deep reinforcement learning for building HVAC control." In Proceedings of the 54th Annual Design Automation Conference 2017, p. 22. ACM, 2017.
8. Zhang, Zhiang, and Khee Poh Lam. "Practical implementation and evaluation of deep reinforcement learning control for a radiant heating system." In Proceedings of the 5th Conference on Systems for Built Environments, pp. 148-157. ACM, 2018.


