# E-Commerce Recommendation Engine

----
## Contents

1. [Background](#background)
1. [Approach](#approach)
1. [Data](#data)
1. [Requirements](#requirements)
1. [Architecture](#architecture)
1. [Cleaning Up](#cleaning-up)
1. [Useful Resources](#useful-resources)

----

## Background

The purpose of this repository is to demonstrate a personalized recommendation engine solution for e-commerce data via Amazon SageMaker Studio. This will give business users a quick path towards a recommendation engine POC. In this demo, we will show how to use SageMaker Studio, Lineage, Model Registry, and Pipelines to build, train, and deploy a personalization model using e-commerce data. 

The overall topics covered in the series of notebook are the following:  

* Setup for using SageMaker
* Basic data cleaning and preprocessing using SageMaker Data Wrangler
* Converting datasets to Protobuf RecordIO format
* Training SageMaker's factorization machines algorithm
* Deploying and getting predictions
* Tracking model artifacts using Lineage
* Registering a model in the model registry
* Building pipeline with SageMaker Pipelines

### What is personalization? What is a recommendation engine?

Personalization, at the highest level, describes how an organization delivers more customized interactions and unique experiences for users. It is a means to meet user expectations by delivering the right experience at the right time, and the right place. 

The concept of personalization (although not new), now offers organizations the ability to improve brand loyalty, grow revenue, and increase efficiency by using data to create a more sophisticated and customized customer experience. With traditional approaches to personalization, they’re typically based on broad segment of users, not tailored for every individual user and therefore the recommendations often times miss the mark.

Machine learning provides a scalable way to deliver unique experiences to individuals based on their behavior and inferred preferences, rather than generic segments of users. Machine learning can help by processing customer data, and selecting the right algorithms to dynamically present the most relevant products or content to each and every user at the right time.

A recommendation engine is a solution that delivers personalized recommendations to each user.

## Approach

----

Our data is transactional data containing invoice numbers, stock codes (i.e. item IDs), product descriptions, quantity sold, date of purchase, price, customer IDs, and country of the customer. Since we do not have customer rating in our data, we will use the quantity purchased as a proxy for rating. In other words, we will try to predict quantities and if we predict a higher quantity for certain customer-product combination, we will take that to mean a higher likelihood that the customer will purchase a given product. If we did have a customer rating, we could use that for our prediction instead. For example, if we had thumbs-up or thumbs-down ratings then we could use factorization machines for binary classification.

For this example, we aggregate the purchasing behavior over the entire year, meaning that we sum how many times each customer bought each product over the timespan. However another approach would be to account for the timing of the purchases by including information from the timestamp. 

To build our model, we use SageMaker built-in algorithm called factorization machines in order to predict which products to recommend to different users. A factorization machine is a general-purpose supervised learning algorithm that you can use for both classification and regression tasks. It is an extension of a linear model that is designed to capture interactions between features within high dimensional sparse datasets efficiently. For recommendation engines, we typically only have high dimensional sparse data since most customers only buy a tiny subset of all the products offered. Factorization machines are particularly well-suited for this kind of a dataset. 

__Overview of the approach:__
1. Read in our data, then do some data cleaning and feature engineering to prepare the data for model training. 
1. Train our model using SageMaker’s built-in factorization machines algorithm
1. Use Amazon SageMaker Lineage Tracking to track several artifacts from our model
1. reate a model package group to register our model in the SageMaker model registry
1. Use SageMaker Pipelines to create a pipeline which will contain all of the aforementioned steps and allow us to automate them 

## Data

The data comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Retail) and contains online retail sales transactions.

Citation:
Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).

## Requirements

You will need an AWS account to use this solution. Sign up for an [account](https://aws.amazon.com/) before you proceed. 

You will also need to have permission to use [Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio.html) and to create all the resources detailed in the [architecture section](#architecture). All AWS permissions can be managed through [AWS IAM](https://aws.amazon.com/iam/). Admin users will have the required permissions, but please contact your account's AWS administrator if your user account doesn't have the required permissions.

To run this notebook under your own AWS account, you'll need to first create an S3 bucket and change the Amazon S3 locations within the code. For data, you have the option to use the same pregenerated data set used in this notebook found in the data folder, recreate the data using the initial generation code and specified changes or replace the data with your own data instead.

## Architecture

As part of the solution, the following services are used:

* [Amazon S3](https://aws.amazon.com/s3/): Used to store datasets.
* [Amazon SageMaker Studio Notebooks](https://aws.amazon.com/sagemaker/): Used to preprocess and visualize the data, and to train model.
* [Amazon SageMaker Endpoint](https://aws.amazon.com/sagemaker/): Used to deploy the trained model.



## Cleaning Up

When you've finished with this solution, make sure that you delete all unwanted AWS resources. 

**Caution**: You need to manually delete any extra resources that you may have created in this notebook. Some examples include, extra Amazon S3 buckets (to the solution's default bucket) and extra Amazon SageMaker endpoints (using a custom name).


----
## Useful Resources

* Personalized recommendation demo: https://www.youtube.com/watch?v=o_p_8HXh0tY
* [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
* [Amazon SageMaker Python SDK Documentation](https://sagemaker.readthedocs.io/en/stable/)






