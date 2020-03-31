# Network compression using RL

## What is network compression?

Network compression is the process of reducing the size of a trained network, either by removing certain layers or by shrinking layers, while maintaining performance. This notebook implements a version of network compression using reinforcement learning algorithm similar to the one proposed in [1].

[1] [Ashok, Anubhav, Nicholas Rhinehart, Fares Beainy, and Kris M. Kitani. "N2N learning: network to network compression via policy gradient reinforcement learning." arXiv preprint arXiv:1709.06030 (2017)]([https://arxiv.org/abs/1709.06030]).

## This Example

In this example the network compression notebook uses a Sagemaker docker image containing Ray, TensorFlow and OpenAI Gym. The network modification module is
treated as a simulation where the actions produced by reinforcement learning algorithm (remove, shrink, etc.) can be run. The notebook has defined a set of actions for each module. It
demonstrates how one can use the SageMaker Python SDK `script` mode with a `Tensorflow+Ray+Gym` container. You can run
`rl_network_compression_a3c_ray_tensorflow_NetworkCompressionEnv.ipynb` from a SageMaker notebook instance. 
This package relies on the SageMaker TensorFlow base RL container that has Ray installed.

## Summary

The sample notebook demonstrates how to:

 1. Formulate network comrpession as a gym environment.
 2. Train the network compression system using Ray in a distributed fasion. 
 3. Produce and download the checkpoints of best compressed models from S3.
