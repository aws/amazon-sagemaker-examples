# Amazon SageMaker Examples

### Advanced Amazon SageMaker Functionality

These examples that showcase unique functionality available in Amazon SageMaker.  They cover a broad range of topics and will utilize a variety of methods, but aim to provide the user with sufficient insight or inspiration to develop within Amazon SageMaker.

- [Data Distribution Types](data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- [Encrypting Your Data](handling_kms_encrypted_data) shows how to use Server Side KMS encrypted data with Amazon SageMaker training. The IAM role used for S3 access needs to have permissions to encrypt and decrypt data with the KMS key.
- [Using Parquet Data](parquet_to_recordio_protobuf) shows how to bring [Parquet](https://parquet.apache.org/) data sitting in S3 into an Amazon SageMaker Notebook and convert it into the recordIO-protobuf format that many SageMaker algorithms consume. 
- [Connecting to Redshift](working_with_redshift_data) demonstrates how to copy data from Redshift to S3 and vice-versa without leaving Amazon SageMaker Notebooks.
- [Bring Your Own XGBoost Model](xgboost_bring_your_own_model) shows how to use Amazon SageMaker Algorithms containers to bring a pre-trained model to a realtime hosted endpoint without ever needing to think about REST APIs.
- [Bring Your Own k-means Model](kmeans_bring_your_own_model) shows how to take a model that's been fit elsewhere and use Amazon SageMaker Algorithms containers to host it.
- [Installing the R Kernel](install_r_kernel) shows how to install the R kernel into an Amazon SageMaker Notebook Instance.
- [Bring Your Own R Algorithm](r_bring_your_own) shows how to bring your own algorithm container to Amazon SageMaker using the R language.
- [Bring Your Own scikit Algorithm](scikit_bring_your_own) provides a detailed walkthrough on how to package a scikit learn algorithm for training and production-ready hosting.
- [Bring Your Own MXNet Model](mxnet_mnist_byom) shows how to bring a model trained anywhere using MXNet into Amazon SageMaker
- [Bring Your Own TensorFlow Model](tensorflow_iris_byom) shows how to bring a model trained anywhere using TensorFlow into Amazon SageMaker