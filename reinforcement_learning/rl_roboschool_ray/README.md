# Roboschool agents training with Ray and TensorFlow on Amazon SageMaker 

Roboschool is an [open source](https://github.com/openai/roboschool/tree/master/roboschool) physics simulator that is commonly used to train RL policies for robotic systems.  Roboschool defines a [variety](https://github.com/openai/roboschool/blob/master/roboschool/__init__.py) of Gym environments that correspond to different robotics problems. Here we're highlighting a few of them at varying levels of difficulty:

- Reacher (easy) - a very simple robot with just 2 joints reaches for a target
- Hopper (medium) - a simple robot with one leg and a foot learns to hop down a track
- Humanoid (difficult) - a complex 3D robot with two arms, two legs, etc. learns to balance without falling over and then to run on a track  

The simpler problems train faster with less computational resources. The more complex problems are more fun.  

In these examples, we demonstrate:
- Vertical scaling of RL training (single node, multiple CPU cores or GPUs)
- Horizontal scaling of RL training across multiple nodes (CPU or GPU)
- Use of SageMaker's Automatic Model Tuning functionality to optimize the training of an RL model, using the Roboschool environment.

## Contents

* `rl_roboschool_ray.ipynb`: Scaling RL training across multiple CPU cores (vertical scaling)
* `rl_roboschool_ray_automatic_model_tuning.ipynb`: Shows how to use SageMaker's Automatic Model Tuner to optimize hyperparameters
* `rl_roboschool_ray_distributed.ipynb`: Scaling RL training across multiple instances, including heterogeneous GPU and CPU clusters
* `Dockerfile`: Dockerfile building the container with Roboschool, Ray and their dependencies by using SageMaker's RL tensorflow container as base.
* `src/`
  * `train-hopper.py`: PPO config for training RoboschoolHopper-v1
  * `train-humanoid.py`: PPO config for training RoboschoolHumanoid-v1
  * `train-reacher.py`: PPO config for training RoboschoolReacher-v1


