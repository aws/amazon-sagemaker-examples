# Amazon SageMaker Studio Container Build CLI

This repository contains example notebooks that show how to utilize the [Amazon SageMaker Studio Build CLI](https://pypi.org/project/sagemaker-studio-image-build/). Full usage instructions can be found at the link provided. 

The SageMaker Studio Image Build CLI offers data scientists and developers the ability to build SageMaker compatible docker images directly from their Amazon SageMaker Studio environments without the need to setup and connect to secondary docker build environments. Using the simple CLI, customers now have the ability to easily create container images directly from Amazon SageMaker Studio. 


## XGBoost Bring Your Own 


- [xgboost_bring_your_own](/xgboost_bring_your_own): This example walks you through the lifecycle of processing some data with SageMaker Processing, training a model using a custom XGBoost container and using SageMaker Batch Transform to generate inferences in batch mode. You will use the studio-image-build CLI to build the XGBoost image, deploy the image to Elastic Container Registry (ECR), and pass the container to the training instance on SageMaker. The [Batch_Transform_BYO_XGB.ipynb notebook](./xgboost_bring_your_own/Batch_Transform_BYO_XGB.ipynb) covers the steps in detail. The XGBoost folder contains the training and inference scripts and Dockerfile refers to the Dockerfile that the studio-image-build CLI will use to build the image.

