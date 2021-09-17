# Amazon SageMaker Studio Container Build CLI

This repository contains example notebooks that show how to utilize the [Amazon SageMaker Studio Build CLI](https://pypi.org/project/sagemaker-studio-image-build/). Full usage instructions can be found at the link provided. 

The SageMaker Studio Image Build CLI offers data scientists and developers the ability to build SageMaker compatible docker images directly from their Amazon SageMaker Studio environments without the need to setup and connect to secondary docker build environments. Using the simple CLI, customers now have the ability to easily create container images directly from Amazon SageMaker Studio. 

## Examples

## Tensorflow Bring Your Own

These examples provide quick walkthroughs to get you up and running with the labeling job workflow for Amazon SageMaker Ground Truth.

- [tensorflow_bring_your_own](/tensorflow_bring_your_own): This example is based on an [existing](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/tensorflow_bring_your_own) example notebook targeted to run on SageMaker Notebook Instances.  However, this example has been modified to run on Amazon SageMaker Studio using the new CLI to locally build a docker image. We show how to package a custom TensorFlow container using the CLI with a Python example which works with the CIFAR-10 dataset and uses TensorFlow Serving for inference. A Dockerfile is included along with code, in the cifar10 directory, that is used for building the image. The utils folder contains a script for converting CIFAR-10 records into TFRecords. The [tensorflow_bring_your_own.ipynb notebook](./tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb) notebooks covers each step in detail. 


## XGBoost Bring Your Own 


- [xgboost_bring_your_own](/xgboost_bring_your_own): This example walks you through the lifecycle of processing some data with SageMaker Processing, training a model using a custom XGBoost container and using SageMaker Batch Transform to generate inferences in batch mode. You will use the studio-image-build CLI to build the XGBoost image, deploy the image to Elastic Container Registry (ECR), and pass the container to the training instance on SageMaker. The [Batch_Transform_BYO_XGB.ipynb notebook](./xgboost_bring_your_own/Batch_Transform_BYO_XGB.ipynb) covers the steps in detail. The XGBoost folder contains the training and inference scripts and Dockerfile refers to the Dockerfile that the studio-image-build CLI will use to build the image.

