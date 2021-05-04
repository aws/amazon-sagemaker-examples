# E-Commerce Personalization
---
## Contents

1. [Background](#background)
1. [Approach](#approach)
1. [Data](#data)
1. [Requirements](#requirements)
1. [Architecture](#architecture)
1. [Cleaning Up](#cleaningUp)


---

## Background
The purpose of this repository is to demonstrate a personalized recommendation engine solution for e-commerce data via Amazon SageMaker Studio. This will give business users a quick path towards a PrM POC. In this notebook, we focus on preprocessing engine sensor data before feature engineering and building an initial model leveraging SageMaker's algorithms. The overall topics covered in this notebook are the following:  


* Setup for using SageMaker
* Basic data cleaning, analysis and preprocessing
* Converting datasets to format used by the Amazon SageMaker algorithms and uploading to S3 
* Training SageMaker's factorization machines algorithm
* Deplying and getting predictions
* Tracking model artifacts using Lineage
* Registering a model
* Creating model registry
* Building pipeline
<br>


## Files:
* __personalization_demo__ - most current version. Use this one 
* __export_dw_job.ipynb__ - Export from Data Wrangler Job. Builds pre-processed data
* __aws_personalization.ipynb__ - Picks up the output of the Data Wrangler job and manually builds the personalization model, including predictions at the end
* __full_pipeline.ipynb__ - entire pipeline build, with steps for Data Wrangler preprocessing, train-test split, model building, model creation, model registry, and deployment.
* __processing.py__ - processing script for the old notebook version (i.e. personalization_1_FtEng_Old.ipynb)
* __processing_2.py__ - processing script for the new notebook version (i.e. aws_personalization.ipynb)


https://archive.ics.uci.edu/ml/datasets/Online+Retail