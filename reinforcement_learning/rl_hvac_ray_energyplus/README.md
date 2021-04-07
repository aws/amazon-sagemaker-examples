# Optimizing building HVAC with Amazon SageMaker RL

## HVAC and EnergyPlus?

HVAC stands for Heating, Ventilation and Air Conditioning and is responsible for keeping us warm and comfortable indoors. HVAC takes up a whopping 50% of the energy in a building [1, 2] and accounts for 10% of global electricity use [3], and several control system optimizations have been proposed to reduce this energy use while ensuring thermal comfort [4, 5, 6].

In a modern building, data such as weather, occupancy and equipment use are collected routinely and can be used to optimize HVAC energy use. Reinforcement Learning (RL) is a good fit as it can learn patterns in the data and identify strategies to control the system so as to reduce energy. Several recent research efforts have shown that RL can reduce HVAC energy consumption by 15-20% [7, 8, 9].

As training an RL algorithm in a real HVAC system can take time to converge as well as potentially lead to hazardous settings as the agent explores its state space, we turn to a simulator to train the agent. [EnergyPlus](https://energyplus.net/) is an open source, state of the art HVAC simulator from the US Department of Energy. We use a simple example with this simulator to showcase how we can train an RL model easily with Amazon SageMaker.

## HVAC control of an office building

Below is the setup we have used for training reinforcement learning for HVAC control:
* **Building**: DOE Commercial Reference Building Medium office, new construction 90.1-2004.
* **Weather file**: San Francisco for year 1999.
* **Simulation days**: Entire year, i.e. 365 days. This number is adjustable.
* **Zones considered**: Core and perimeter for all zones
* **Controls**: Heating and cooling setpoints for all zones. This setting is adjustable from multi-zone control to single control for all zones.
* **Reward**: sum of energy penalty (total energy * penalty coeffient) and mean zone temperature penalty (absolute difference between desired temperature and actual temperature). The penalty ratio between energy and temperature can be adjusted.
* **Algorithm**: Default is APEX_DDPG. We have tested with PPO [10] as well.

We have used threads, queues, and callbacks from EnergyPlus to orchestrate distributed co-simulation between Open AI gym interface and EnergyPlus. Higher number of cores help with faster training. That's why our default choice is *ml.g4dn16xlarge* instance, which has 64 cores. We have found around 50 training iterations are sufficient for reward value to converge.

## Contents

* `train-hvac.ipynb`: Main notebook used for training HVAC policy.
* `Dockerfile`: used to build custom container that Ray RLlib TensorFlow or PyTorch container.
* `source/`  
  * `train-sagemaker-hvac.py`: fixed baseline policy for establishing baseline performance.
  * `eplus`: directory for EnergyPlus Python interface for environments.
  * `medium_office_env.py`: Open AI Gym file that provides interface for training with Ray RLlib.
    * `buildings`: contains EnergyPlus IDF (building model) files
    * `weather`: contains weather files

### Container Versions:

Python: 3.6.4
Ray: 0.8.5
EnergyPlus: 9.3.0 (https://github.com/NREL/EnergyPlus/releases/tag/v9.3.0)

## References

1. Residential Energy Consumption Survey - https://www.eia.gov/consumption/residential/
2. Energy Use in Commercial Buildings - https://www.eia.gov/energyexplained/index.php?page=us\_energy\_commercial  
3. The Future of Cooling - https://www.iea.org/futureofcooling/ 
4. Afram, Abdul, and Farrokh Janabi-Sharifi. "Theory and applications of HVAC control systems–A review of model predictive control (MPC)." Building and Environment 72 (2014): 343-355.
5. Fong, Kwong Fai, Victor Ian Hanby, and Tin-Tai Chow. "System optimization for HVAC energy management using the robust evolutionary algorithm." Applied Thermal Engineering 29, no. 11-12 (2009): 2327-2334.
6. Balaji, Bharathan, Jian Xu, Anthony Nwokafor, Rajesh Gupta, and Yuvraj Agarwal. "Sentinel: Occupancy based HVAC actuation using existing WiFi infrastructure within commercial buildings." In Proceedings of the 11th ACM Conference on Embedded Networked Sensor Systems, p. 17. ACM, 2013.
7. Wei, Tianshu, Yanzhi Wang, and Qi Zhu. "Deep reinforcement learning for building HVAC control." In Proceedings of the 54th Annual Design Automation Conference 2017, p. 22. ACM, 2017.
8. Zhang, Zhiang, and Khee Poh Lam. "Practical implementation and evaluation of deep reinforcement learning control for a radiant heating system." In Proceedings of the 5th Conference on Systems for Built Environments, pp. 148-157. ACM, 2018.
9. Moriyama T., De Magistris G., Tatsubori M., Pham TH., Munawar A., Tachibana R. "Reinforcement Learning Testbed for Power-Consumption Optimization". In: Li L., Hasegawa K., Tanaka S. (eds) Methods and Applications for Modeling and Simulation of Complex Systems. AsiaSim 2018. Communications in Computer and Information Science, vol 946. Springer, Singapore.
10. J Schulman, F Wolski, P Dhariwal, A Radford, O Klimov. "Proximal policy optimization algorithms". arXiv:1707.06347, 2017
