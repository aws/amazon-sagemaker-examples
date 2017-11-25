# Advanced Functionality

This directory includes examples which showcase unique functionality available in Amazon SageMaker.  Examples cover a broad range of topics and will utilize a variety of methods, but aim to provide the user with sufficient insight or inspiration to develop within Amazon SageMaker.

Example Notebooks include:
- *data_distribution_types*: Showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- *install_r_kernel*: A quick introduction to getting R installed and running within Amazon SageMaker Notebook Instances.
- *kmeans_bring_your_own_model*: How to use Amazon SageMaker Algorithms containers to bring a pre-trained model to a realtime hosted endpoint without ever needing to think about REST APIs.
- *r_bring_your_own*: How to containerize an R algorithm using Docker and plumber for hosting so that it can be used in Amazon SageMaker's managed training and realtime hosting.
- *xgboost_bring_your_own_model*: How to use Amazon SageMaker Algorithms containers to bring a pre-trained model to a realtime hosted endpoint without ever needing to think about REST APIs.
- *handling_kms_encrypted_data.ipynb*: How to use Server Side KMS encrypted data with Amazon SageMaker training works. The IAM role used for S3 access needs to have permissions to encrypt and decrypt data with the KMS key.
- *parquet_to_recordio_protobuf.ipynb*: How to convert Parquet data format into the recordIO-protobuf format that many SageMaker algorithms consume. 
- *working_with_redshift_data.ipynb*: Demonstrates how to copy data from Redshift to S3 and vice-versa. 
