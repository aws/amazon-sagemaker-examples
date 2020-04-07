# Amazon SageMaker Examples

## AWS Marketplace

This repository contains example notebooks that show how to use [algorithms and model packages from AWS Marketplace for machine learning](https://aws.amazon.com/marketplace/search/results?page=1&filters=fulfillment_options&fulfillment_options=SAGEMAKER)

To know more about algorithms and model packages from AWS Marketplace, see [documentation](https://docs.aws.amazon.com/marketplace/latest/userguide/machine-learning-products.html)

#### Create algorithms/model packages for listing in AWS Marketplace for machine learning.

This example notebook shows you how to package a model-package/algorithm for listing in AWS Marketplace for machine learning.

- [Creating Algorithm and Model Package - Listing on AWS Marketplace](creating_marketplace_products) provides a detailed walkthrough on how to package a scikit learn algorithm to create SageMaker Algorithm and SageMaker Model Package entities that can be used with the enhanced SageMaker Train/Transform/Hosting/Tuning APIs and listed on AWS Marketplace.

#### Use algorithms and model packages from AWS Marketplace for machine learning.

These examples show you how to use model-packages and algorithms from AWS Marketplace for machine learning.

- [Using Algorithms](using_algorithms)
	- [Using Algorithm From AWS Marketplace](using_algorithms/amazon_demo_product) provides a detailed walkthrough on how to use Algorithm with the enhanced SageMaker Train/Transform/Hosting/Tuning APIs by choosing a canonical product listed on AWS Marketplace.
	- [Using AutoML algorithm](using_algorithms/automl) provides a detailed walkthrough on how to use AutoML algorithm from AWS Marketplace.

- [Using Model Packages](using_model_packages)
	- [Using Model Packages From AWS Marketplace](using_model_packages/generic_sample_notebook) is a generic notebook which provides sample code snippets you can modify and use for performing inference on Model Packages from AWS Marketplace, using Amazon SageMaker.
	- [Using Amazon Demo product From AWS Marketplace](using_model_packages/amazon_demo_product) provides a detailed walkthrough on how to use Model Package entities with the enhanced SageMaker Transform/Hosting APIs by choosing a canonical product listed on AWS Marketplace.
	- [Using models for extracting vehicle metadata](using_model_packages/auto_insurance) provides a detailed walkthrough on how to use pre-trained models from AWS Marketplace for extracting metadata for a sample use-case of auto-insurance claim processing.
	- [Using models for identifying non-compliance at a workplace](using_model_packages/improving_industrial_workplace_safety) provides a detailed walkthrough on how to use pre-trained models from AWS Marketplace for extracting metadata for a sample use-case of generating summary reports for identifying non-compliance at a construction/industrial workplace.
	- [Extracting insights from your credit card statements](using_model_packages/financial_transaction_processing) provides a detailed walkthrough on how to use pre-trained models from AWS Marketplace for efficiently processing financial transaction logs.

## FAQ

*What do I need in order to get started?*

- The quickest setup to run example notebooks includes:
  - An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
  - Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
  - An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
  - An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)
  - [AWS Marketplace Subscription](https://aws.amazon.com/marketplace/help/200799470#topic1) to the algorithm/model you wish to use.

