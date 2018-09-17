# Bring Your Own Pipe-mode Algorithm

This folder contains one notebook and a few helper files:

*train.py* is a python script which defines a few functions that together implement a rudimentary mechanism via which to train using SageMaker Training's Pipe-mode.

*Dockerfile:* is the necessary configuration for building a docker container that calls the `train.py` script.

*pipe_bring_your_own.ipynb:* is a notebook that calls the custom container once built and pushed into ECR.
