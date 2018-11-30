# DeepRacer with Amazon SageMaker RL and AWS RoboMaker

This folder contains examples of how to use RL to train an autonomous race car.


## Contents

* `rl_deepracer_clippedppo_coach_tensorflow_robomaker.ipynb`: notebook for training autonomous race car.


* `src/`
  * `training_worker.py`: Main entrypoint for starting distributed training job
  * `robomaker/presets/deepracer.py`: Preset (configuration) for DeepRacer
  * `robomaker/environments/deepracer_env.py`: Gym environment file for DeepRacer
  * `markov/`: Helper files for S3 upload/download
