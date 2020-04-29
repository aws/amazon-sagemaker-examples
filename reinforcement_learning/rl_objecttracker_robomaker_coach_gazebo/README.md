# Object Tracker with Amazon SageMaker RL and AWS RoboMaker

This folder contains examples of how to use RL to train a TurtleBot object tracker using Amazon SageMaker Reinforcement Learning and AWS RoboMaker.


## Contents

* `rl_objecttracker_clippedppo_coach_tensorflow_robomaker.ipynb`: notebook for training an object tracker.


* `src/`
  * `training_worker.py`: Main entrypoint for starting distributed training job
  * `robomaker/presets/object_tracker.py`: Preset (configuration) for object tracker
  * `robomaker/environments/object_tracker_env.py`: Gym environment file for object tracker
  * `markov/`: Helper files for S3 upload/download
