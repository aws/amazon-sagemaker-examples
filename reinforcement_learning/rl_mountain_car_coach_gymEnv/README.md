# Mountain Car with Amazon SageMaker RL

Mountain Car is a classic control Reinforcement Learning problem that was first introduced by A. Moore in 1991 [1]. An under-powered car is tasked with climbing a steep mountain, and is only successful when it reaches the top. Luckily there's another mountain on the opposite side which can be used to gain momentum, and launch the car to the peak. It can be tricky to find this optimal solution due to the sparsity of the reward. Complex exploration strategies can be used to incentivise exploration of the mountain, but to keep things simple in this example we extend the amount of time in each episode from Open AI Gym's default of 200 environment steps to 10,000 steps, showing how to customise environments. We consider two variants in this example: PatientMountainCar for discrete actions and PatientContinuousMountainCar for continuous actions.

## Contents

* `rl_mountain_car_coach_gym.ipynb`: notebook used for training Mountain Car policy.
* `src/`
  * `patient_envs.py`: custom environments defined here.
  * `train-coach.py`: launcher for coach training
  * `preset-mountain-car-continuous-clipped-ppo.py`: coach preset for Clipped PPO.
  * `preset-mountain-car-dqn.py`: coach preset for DQN.
