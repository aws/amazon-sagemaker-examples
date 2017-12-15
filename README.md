# Amazon SageMaker Examples

This repository contains example notebooks that show how to apply machine learning and deep learning in [Amazon SageMaker](https://aws.amazon.com/sagemaker)

## Examples

### Introduction to Applying Machine Learning

These examples provide a gentle introduction to machine learning concepts as they are applied in practical use cases across a variety of sectors.

- [Targeted Direct Marketing](introduction_to_applying_machine_learning/xgboost_direct_marketing) predicts potential customers that are most likely to convert based on customer and aggregate level metrics, using Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost).
- [Predicting Customer Churn](introduction_to_applying_machine_learning/xgboost_customer_churn) uses customer interaction and service usage data to find those most likely to churn, and then walks through the cost/benefit trade-offs of providing retention incentives.  This uses Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost) to create a highly predictive model.
- [Time-series Forecasting](introduction_to_applying_machine_learning/linear_time_series_forecast) generates a forecast for topline product demand using Amazon SageMaker's Linear Learner algorithm.
- [Cancer Prediction](introduction_to_applying_machine_learning/breast_cancer_prediction) predicts Breast Cancer based on features derived from images, using SageMaker's Linear Learner.
- [Ensembling](introduction_to_applying_machine_learning/ensemble_modeling) predicts income using two Amazon SageMaker models to show the advantages in ensembling.
- [Video Game Sales](introduction_to_applying_machine_learning/video_game_sales) develops a binary prediction model for the success of video games based on review scores.

### Introduction to Amazon Algorithms

These examples provide quick walkthroughs to get you up and running with Amazon SageMaker's custom developed algorithms.  Most of these algorithms can train on distributed hardware, scale incredibly well, and are faster and cheaper than popular alternatives.

- [k-means](sagemaker-python-sdk/1P_kmeans_highlevel) is our introductory example for Amazon SageMaker.  It walks through the process of clustering MNIST images of handwritten digits using Amazon SageMaker k-means.
- [Factorization Machines](introduction_to_amazon_algorithms/factorization_machines_mnist) showcases Amazon SageMaker's implementation of the algorithm to predict whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier.
- [Latent Dirichlet Allocation (LDA)](introduction_to_amazon_algorithms/lda_topic_modeling) introduces topic modeling using Amazon SageMaker Latent Dirichlet Allocation (LDA) on a synthetic dataset.
- [Linear Learner](introduction_to_amazon_algorithms/linear_learner_mnist) predicts whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier from Amazon SageMaker Linear Learner.
- [Neural Topic Model (NTM)](introduction_to_amazon_algorithms/ntm_synthetic) uses Amazon SageMaker Neural Topic Model (NTM) to uncover topics in documents from a synthetic data source, where topic distributions are known.
- [Principal Components Analysis (PCA)](introduction_to_amazon_algorithms/pca_mnist) uses Amazon SageMaker PCA to calculate eigendigits from MNIST.
- [Seq2Seq](introduction_to_amazon_algorithms/seq2seq) uses the Amazon SageMaker Seq2Seq algorithm that's built on top of [Sockeye](https://github.com/awslabs/sockeye), which is a sequence-to-sequence framework for Neural Machine Translation based on MXNet.  Seq2Seq implements state-of-the-art encoder-decoder architectures which can also be used for tasks like Abstractive Summarization in addition to Machine Translation.  This notebook shows translation from English to German text.
- [Image Classification](introduction_to_amazon_algorithms/imageclassification_caltech) includes full training and transfer learning examples of Amazon SageMaker's Image Classification algorithm.  This uses a ResNet deep convolutional neural network to classify images from the caltech dataset.
- [XGBoost for regression](introduction_to_amazon_algorithms/xgboost_abalone) predicts the age of abalone ([Abalone dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html)) using regression from Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost).
- [XGBoost for multi-class classification](introduction_to_amazon_algorithms/xgboost_mnist) uses Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost) to classify handwritten digits from the MNIST dataset as one of the ten digits using a multi-class classifier. Both single machine and distributed use-cases are presented.

### Scientific Details of Algorithms

These examples provide more thorough mathematical treatment on a select group of algorithms.

- [Latent Dirichlet Allocation (LDA)](scientific_details_of_algorithms/lda_topic_modeling) dives into Amazon SageMaker's spectral decomposition approach to LDA.

### Advanced Amazon SageMaker Functionality

These examples that showcase unique functionality available in Amazon SageMaker.  They cover a broad range of topics and will utilize a variety of methods, but aim to provide the user with sufficient insight or inspiration to develop within Amazon SageMaker.

- [Data Distribution Types](advanced_functionality/data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- [Encrypting Your Data](advanced_functionality/handling_kms_encrypted_data) shows how to use Server Side KMS encrypted data with Amazon SageMaker training. The IAM role used for S3 access needs to have permissions to encrypt and decrypt data with the KMS key.
- [Using Parquet Data](advanced_functionality/parquet_to_recordio_protobuf) shows how to bring [Parquet](https://parquet.apache.org/) data sitting in S3 into an Amazon SageMaker Notebook and convert it into the recordIO-protobuf format that many SageMaker algorithms consume. 
- [Connecting to Redshift](advanced_functionality/working_with_redshift_data) demonstrates how to copy data from Redshift to S3 and vice-versa without leaving Amazon SageMaker Notebooks.
- [Bring Your Own XGBoost Model](advanced_functionality/xgboost_bring_your_own_model) shows how to use Amazon SageMaker Algorithms containers to bring a pre-trained model to a realtime hosted endpoint without ever needing to think about REST APIs.
- [Bring Your Own k-means Model](advanced_functionality/kmeans_bring_your_own_model) shows how to take a model that's been fit elsewhere and use Amazon SageMaker Algorithms containers to host it.
- [Bring Your Own R Algorithm](advanced_functionality/r_bring_your_own) shows how to bring your own algorithm container to Amazon SageMaker using the R language.
- [Installing the R Kernel](advanced_functionality/install_r_kernel) shows how to install the R kernel into an Amazon SageMaker Notebook Instance.
- [Bring Your Own scikit Algorithm](advanced_functionality/scikit_bring_your_own) provides a detailed walkthrough on how to package a scikit learn algorithm for training and production-ready hosting.
- [Bring Your Own TensorFlow Model](advanced_functionality/tensorflow_iris_byom) shows how to bring a model trained anywhere using TensorFlow into Amazon SageMaker

### Amazon SageMaker TensorFlow and MXNet Pre-Built Containers and the Python SDK

These examples focus on the Amazon SageMaker Python SDK which allows you to write idiomatic TensorFlow or MXNet and then train or host in pre-built containers.

- [cifar 10 with MXNet Gluon](sagemaker-python-sdk/mxnet_gluon_cifar10)
- [MNIST with MXNet Gluon](sagemaker-python-sdk/mxnet_gluon_mnist)
- [MNIST with MXNet](sagemaker-python-sdk/mxnet_mnist)
- [Sentiment Analysis with MXNet Gluon](sagemaker-python-sdk/mxnet_gluon_sentiment)
- [TensorFlow Neural Networks with Layers](sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_layers)
- [TensorFlow Networks with Keras](sagemaker-python-sdk/tensorflow_abalone_age_predictor_using_keras)
- [Introduction to Estimators in TensorFlow](sagemaker-python-sdk/tensorflow_iris_dnn_classifier_using_estimators)
- [TensorFlow and TensorBoard](sagemaker-python-sdk/tensorflow_resnet_cifar10_with_tensorboard)
- [Distributed TensorFlow](sagemaker-python-sdk/tensorflow_distributed_mnist)

### Under Development

These Amazon SageMaker examples fully illustrate a concept, but may require some additional configuration on the users part to complete.

- [Bring Your Own MXNet Model](under_development/mxnet_mnist_byom) shows how to bring a model trained anywhere using MXNet into Amazon SageMaker

## FAQ

*What do I need in order to get started?*

- The quickest setup to run example notebooks includes:
  - An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
  - Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
  - An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
  - An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

*Will these examples work outside of Amazon SageMaker Notebook Instances?*

- Although most examples utilize key Amazon SageMaker functionality like distributed, managed training or real-time hosted endpoints, these notebooks can be run outside of Amazon SageMaker Notebook Instances with minimal modification (updating IAM role definition and installing the necessary libraries).

*How do I contribute my own example notebook?*

- Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources.  Please bear with us in the short-term if pull requests take longer than expected or are closed.
