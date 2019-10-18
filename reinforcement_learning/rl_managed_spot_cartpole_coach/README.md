# Simple Cartpole Example with SageMaker Managed Spot Training

The example here is almost the same as Cart-pole Balancing Model with Amazon SageMaker.

This notebook tackles the exact same problem with the same solution, but it has been modified to be able to run using SageMaker Managed Spot infrastructure. SageMaker Managed Spot uses EC2 Spot Instances to run Training at a lower cost.

Please read the original notebook and try it out to gain an understanding of the ML use-case and how it is being solved. We will not delve into that here in this notebook.

The explanations below has been lifted verbatim from Cart-pole Balancing Model with Amazon SageMaker.


## What is cartpole?

This is a classic example of a problem that reinforcement learning can solve.  It's a simulation of balancing a broom upright by balancing it on your hand.  The broom is the "pole" and your hand is replaced with a "cart" moving back and forth on a linear track.  This simplified example works in 2 dimensions, so the cart can only move in a line back and forth, and the pole can only fall forwards or backwards, not to the sides.

## This Example

Sample Cart Pole Example using SageMaker RL with base docker image containing Coach, MxNet and OpenAI Gym. This is a toy example taken from 
[Coach Quick Start Guide](https://github.com/NervanaSystems/coach/blob/master/tutorials/0.%20Quick%20Start%20Guide.ipynb). It demonstrates how you can use the `RLEstimator` from the SageMaker Python SDK in `script` mode.

## Summary

The sample notebook demonstrates how to:

 1. Train a toy cart pole model from a notebook and python SDK/script mode
 2. Visualize `gifs` generated during training. 
 3. Move these back and forth from S3 (while running in `SageMaker` mode)
 4. Checkpoint trained model
 5. Run inference using checkpointed models
