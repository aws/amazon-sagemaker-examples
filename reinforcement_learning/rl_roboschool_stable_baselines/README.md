# Roboschool simulations training with stable baselines on Amazon SageMaker 

Roboschool is an [open source](https://github.com/openai/roboschool/tree/master/roboschool) physics simulator that is commonly used to train RL policies for robotic systems.  Roboschool defines a [variety](https://github.com/openai/roboschool/blob/master/roboschool/__init__.py) of Gym environments that correspond to different robotics problems. One of them is **HalfCheetah** which is a two-legged robot, restricted to a vertical plane, meaning it can only run forward or backward.

In this notebook example, we will make **HalfCheetah** learn to walk using the [stable-baselines](https://stable-baselines.readthedocs.io/en/master/) a set of improved implementations of Reinforcement Learning (RL) algorithms based on [OpenAI Baselines](https://github.com/openai/baselines). 


## Contents

* `rl_roboschool_stable_baselines.ipynb`: Notebook demonstrating the code to make *HalfCheetah* learn to walk.
* `Dockerfile`: Dockerfile building the container with Roboschool, OpenMPI, stable-baselines and their dependencies by using SageMaker's RL tensorflow container as base.
* `src/`
  * `preset-half-cheetah.py`: Preset for HalfCheetah distributed training with Stable-Baselines PPI1. 
  * `train_stable_baselines.py`: Training Stable-Baselines launcher script.
* `resources`: Files required as part of docker build.
* `examples`:
  * `robo_half_cheetah_10x_40min.mp4`: Output RL video for model trained using the `rl_roboschool_stable_baselines.ipynb` notebook with `10 ml.c4.xlarge` instances and `num_timesteps` as `1e7`

