# Amazon SageMaker Feature Store

## Introduction to Feature Store
In `feature_store_introduction.ipynb` we demonstrate how to get started with Feature Store, create feature groups, and ingest data into them.

This notebook requires these data sets in `./data/`:

* `feature_store_introduction_customer.csv`
* `feature_store_introduction_orders.csv`

This notebook requires these images in `./images/`:

* `feature-store-policy.png`
* `feature_store_data_ingest.svg`.

## Feature Store Encryption with KMS key
In `feature_store_kms_encryption.ipynb` we demonstrate how to encrypt data in your online or offline store using KMS key and how to verify that your KMS key is being used for data encryption. 

This notebook requires these data sets in `./data/`:
* `feature_store_introduction_customer.csv`
* `feature_store_introduction_orders.csv`

This notebook requires these images in `./images/`:
* `cloud-trails.png`
* `s3-sse-enabled.png`

## Fraud Detection with Feature Store
For an advanced example on how to use Feature Store for a Fraud Detection use-case, see [Fraud Detection with Feature Store](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-featurestore/sagemaker_featurestore_fraud_detection_python_sdk.html), and it's associated notebook, `sagemaker_featurestore_fraud_detection_python_sdk.ipynb`.

## Developer Guide
For detailed information about Feature Store, see the [Feature Store Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html).  
