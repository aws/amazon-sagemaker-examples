# Amazon SageMaker Feature Store
## Using Health Record Data to Detect Heart Failure with SageMaker Feature Store

## Contents
1. [Background](#Background)
1. [Prerequisites](#Prereqs)
1. [Data](#Data)
1. [Approach](#Approach)
1. [Clean Up](#Clean-Up)
1. [Other Resources](#Other-Resources)

---

# Background

SageMaker Feature Store makes it easy to create and manage curated features for machine learning (ML) development. It serves as the single source of truth to store, retrieve, remove, track, share, discover, and control access to features. SageMaker Feature Store enables data ingestion via a high TPS API and data consumption via the online and offline stores.

In this notebook we use SageMaker Feature Store to prepare and store features to train a heart failure detection model using medical record data. This notebook demonstrates how the dataset can be ingested into the Feature Store, queried to create a training dataset, and quickly accessed during inference. We also see how to integrate SageMaker Feature Store with SageMaker Data Wrangler and SageMaker Pipelines to process, store and use features in machine learning development.


# Prereqs

The following IAM policies need to be attached to the SageMaker execution role that you use to run this notebook:

- AmazonSageMakerFullAccess
- AmazonSageMakerFeatureStoreAccess
- AmazonS3FullAccess

Note that the AmazonS3FullAccess policy is not attached to your role by default if you choose to create a new role when you start your SageMaker Studio instance. If you don't see the required policies above are listed under Policy name, you can go to the IAM console, find your role, choose Attach Policies under Permissions, find the policies you are missing from the list, then choose Attach policy. For more information, see the [Developer Guide: SageMaker Roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)


# Data

This notebook uses the publicly available [Heart failure clinical records Data Set](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) that can be downloaded from the UCI machine learning Repository, as described in the notebook. The data set contains medical record information for a small sample of heart failure patients, including demographic, diagnostic and laboratory test data.

**heart_failure_clinical_records_dataset.csv**  

The dataset contains one table with thirteen (13) columns:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- (target)death event: if the patient deceased during the follow-up period (boolean)


 # Approach
 
![architecture diagram](/fs1.PNG)
 
First, we'll prepare the data for Feature Store, create a Feature Group and then ingest our data in to the Feature Group. Our features will be available in the offline feature store within minutes. We then use the feature store to build a training dataset, fit a simple model and return predictions.

# Clean Up

In order to prevent ongoing charges to your AWS account, clean up any resources we spun up during this tutorial.


# Other Resources
  
- SageMaker Feature Store Introductory Blog: [New – Store, Discover, and Share Machine Learning Features with Amazon SageMaker Feature Store](https://aws.amazon.com/blogs/aws/new-store-discover-and-share-machine-learning-features-with-amazon-sagemaker-feature-store/?sc_icampaign=launch_sagemaker-feature-store_reinvent20&sc_ichannel=ha&sc_icontent=awssm-6216&sc_iplace=ribbon&trk=ha_awssm-6216)