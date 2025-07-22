# Amazon SageMaker Pipelines
## Training and deploying a text classification model using Amazon SageMaker Pipelines

## Contents
1. [Background](#Background)
2. [Prerequisites](#Prereqs)
3. [Data](#Data)
4. [Approach](#Approach)
5. [Other Resources](#Other-Resources)

---

# Background

Amazon SageMaker Pipelines makes it easy for data scientists and engineers to build, automate, and scale end-to-end machine learning workflows. Machine learning workflows are complex, requiring iteration and experimentation across each step of the machine learning process, such as exploring and preparing data, experimenting with different algorithms, training and turning models, and deploying models to production. Developing and managing these workflows can take weeks or months of coding and manually managing workflow dependencies can become complex. With Amazon SageMaker Pipelines, data science teams have an easy-to-use continuous integration and continuous delivery (CI/CD) service that simplifies the development and management of machine learning workflows at scale.

In this notebook we use SageMaker Pipelines to train and deploy a text classification model to predict e-commerce product ratings based on customers’ product reviews. We’ll use BlazingText, a SageMaker built-in algorithm, to minimize the amount of effort required to train and deploy the model. BlazingText provides highly optimized implementations of Word2vec and text classification algorithms.

# Prereqs

You will need an AWS account to use this solution. Sign up for an [account](https://aws.amazon.com/) before you proceed. 

You will also need to have permission to use [Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio.html). All AWS permissions can be managed through [AWS IAM](https://aws.amazon.com/iam/). Admin users will have the required permissions, but please contact your account's AWS administrator if your user account doesn't have the required permissions.

# Data

To train the model, we’ll use a sample of data containing e-commerce reviews and associated product ratings. Our pipeline will start with processing the data for model training and will proceed with model training, evaluation, registry and deployment. The Women’s E-Commerce Clothing Clothing Reviews dataset has been made available under a Creative Commons license. A copy of the dataset has been saved in a sample data Amazon S3 bucket. In the first section of the notebook, we’ll walk through how to download the data and get started with building the ML workflow as a SageMaker pipeline.

# Approach

Our ML workflow will be built in the following SageMaker pipeline steps:
* Data processing step - in this step we use a scikit-learn processor to process the training data by cleaning up the review text (eg. remove punctuation and convert to lower case), rebalancing the dataset, creating review categories and generating the training, testing and validation datasets
* Model training step - in this step we create a SageMaker estimator and specify model training hyperparameters and the location of training and validation data
* Create model step - in the create model step we pass the model data from the training step
* Deploy model step - the deploy model step uses a scikit-learn processor to deploy the trained model
* Register model step - in the final model step we submit the trained model to the model registry. We can optionally configure this step to require manual approval before submission.

# Other Resources

For additional SageMaker Pipelines examples, see [Orchestrating Jobs with Amazon SageMaker Model Building Pipelines](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform.html) or the related [GitHub repo](https://github.com/aws/amazon-sagemaker-examples/tree/master/sagemaker-pipelines).